"""Triton kernels for :mod:`schnetpack.nn.ops.radial_filter` (imported only when Triton
is).

Both kernels take the pairs in distance order (``order``), in tiles of ``BLOCK_P`` pairs
x ``BLOCK_C`` output features (``FORWARD_BLOCK_C`` in the forward kernel). A tile loops
over the rows of ``W^T`` within ``half_band`` of its pairs' nearest Gaussian centers;
sorted, that is a few rows per tile. The forward kernel writes each output row once. The gradient kernel sums its
tile's pairs per row and adds the result into ``dW^T`` with atomics, so the summation
order, and with it the last bits of the weight gradient, varies between runs.

In float32 the rows are taken ``BLOCK_K`` at a time as a ``tl.dot`` in IEEE precision
(no TF32, whatever the matmul precision setting); float64 goes row by row.
"""

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = ["gaussian_rbf_linear_triton", "gaussian_rbf_linear_grad_triton", "supported"]

BLOCK_P, BLOCK_C, BLOCK_K = 64, 64, 16
# The forward kernel's tiles: 64 x 128 with 8 warps wrote GPFF's filters (3072
# outputs, 164k and 314k pairs) in 1.5 and 2.9 ms against 2.2 and 4.3 ms with the
# gradient kernel's 64 x 64 and 4 warps, with the same result to the bit (H100 PCIe;
# sandbox/speedup/painn/probe_banded_tiles.py).
FORWARD_BLOCK_C, FORWARD_WARPS = 128, 8


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Inputs the kernels handle; everything else takes the PyTorch path."""
    return (
        x.is_cuda
        and x.dtype == weight.dtype
        and x.dtype in (torch.float32, torch.float64)
    )


@triton.jit
def _tile_band(
    x_ptr, order_ptr, n_pairs, n_rbf, start, spacing, half_band, BLOCK_P: tl.constexpr
):
    """Pair indices, distances and nearest centers of a tile, and the rows it spans."""
    s = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)  # position in distance order
    mask_p = s < n_pairs
    p = tl.load(order_ptr + s, mask=mask_p, other=0)
    d = tl.load(x_ptr + p, mask=mask_p, other=0.0)
    kc = tl.floor((d - start) / spacing + 0.5).to(tl.int32)
    k_lo = tl.maximum(tl.min(tl.where(mask_p, kc, 1 << 30), axis=0) - half_band, 0)
    k_hi = tl.minimum(
        tl.max(tl.where(mask_p, kc, -(1 << 30)), axis=0) + half_band, n_rbf - 1
    )
    return p, d, kc, mask_p, k_lo, k_hi


@triton.jit
def _forward_kernel(
    x_ptr,
    order_ptr,
    wt_ptr,
    bias_ptr,
    offsets_ptr,
    coeff_ptr,
    out_ptr,
    n_pairs,
    n_rbf,
    n_out,
    start,
    spacing,
    half_band,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT: tl.constexpr,
):
    p, d, kc, mask_p, k_lo, k_hi = _tile_band(
        x_ptr, order_ptr, n_pairs, n_rbf, start, spacing, half_band, BLOCK_P
    )
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c < n_out
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=d.dtype)
    acc += tl.load(bias_ptr + c, mask=mask_c, other=0.0)[None, :]
    if DOT:
        for k0 in range(k_lo, k_hi + 1, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            mask_k = k <= k_hi
            band = (
                mask_p[:, None]
                & mask_k[None, :]
                & (tl.abs(kc[:, None] - k[None, :]) <= half_band)
            )
            diff = (
                d[:, None] - tl.load(offsets_ptr + k, mask=mask_k, other=0.0)[None, :]
            )
            cf = tl.load(coeff_ptr + k, mask=mask_k, other=0.0)[None, :]
            phi = tl.where(band, tl.exp(cf * (diff * diff)), 0.0)
            w = tl.load(
                wt_ptr + k[:, None] * n_out + c[None, :],
                mask=mask_k[:, None] & mask_c[None, :],
                other=0.0,
            )
            acc = tl.dot(phi, w, acc, input_precision="ieee")
    else:
        for k in range(k_lo, k_hi + 1):
            band = mask_p & (tl.abs(kc - k) <= half_band)
            diff = d - tl.load(offsets_ptr + k)
            phi = tl.where(band, tl.exp(tl.load(coeff_ptr + k) * (diff * diff)), 0.0)
            w = tl.load(wt_ptr + k * n_out + c, mask=mask_c, other=0.0)
            acc += phi[:, None] * w[None, :]
    mask = mask_p[:, None] & mask_c[None, :]
    tl.store(out_ptr + p[:, None].to(tl.int64) * n_out + c[None, :], acc, mask=mask)


@triton.jit
def _grad_kernel(
    x_ptr,
    order_ptr,
    grad_ptr,
    offsets_ptr,
    coeff_ptr,
    gwt_ptr,
    gbias_ptr,
    n_pairs,
    n_rbf,
    n_out,
    start,
    spacing,
    half_band,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT: tl.constexpr,
):
    p, d, kc, mask_p, k_lo, k_hi = _tile_band(
        x_ptr, order_ptr, n_pairs, n_rbf, start, spacing, half_band, BLOCK_P
    )
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c < n_out
    mask = mask_p[:, None] & mask_c[None, :]
    g = tl.load(
        grad_ptr + p[:, None].to(tl.int64) * n_out + c[None, :], mask=mask, other=0.0
    )
    tl.atomic_add(gbias_ptr + c, tl.sum(g, axis=0), mask=mask_c)
    if DOT:
        for k0 in range(k_lo, k_hi + 1, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            mask_k = k <= k_hi
            band = (
                mask_k[:, None]
                & mask_p[None, :]
                & (tl.abs(k[:, None] - kc[None, :]) <= half_band)
            )
            diff = (
                d[None, :] - tl.load(offsets_ptr + k, mask=mask_k, other=0.0)[:, None]
            )
            cf = tl.load(coeff_ptr + k, mask=mask_k, other=0.0)[:, None]
            phi = tl.where(band, tl.exp(cf * (diff * diff)), 0.0)
            tl.atomic_add(
                gwt_ptr + k[:, None] * n_out + c[None, :],
                tl.dot(phi, g, input_precision="ieee"),
                mask=mask_k[:, None] & mask_c[None, :],
            )
    else:
        for k in range(k_lo, k_hi + 1):
            band = mask_p & (tl.abs(kc - k) <= half_band)
            diff = d - tl.load(offsets_ptr + k)
            phi = tl.where(band, tl.exp(tl.load(coeff_ptr + k) * (diff * diff)), 0.0)
            tl.atomic_add(
                gwt_ptr + k * n_out + c, tl.sum(phi[:, None] * g, axis=0), mask=mask_c
            )


def gaussian_rbf_linear_triton(
    x: torch.Tensor,
    order: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    coeff: torch.Tensor,
    start: float,
    spacing: float,
    half_band: int,
) -> torch.Tensor:
    n_out, n_rbf = weight.shape
    wt = weight.t().contiguous()  # one row per Gaussian
    out = x.new_empty(x.shape[0], n_out)
    if x.shape[0] > 0 and n_out > 0:
        grid = (triton.cdiv(x.shape[0], BLOCK_P), triton.cdiv(n_out, FORWARD_BLOCK_C))
        wrap_triton(_forward_kernel)[grid](
            x.contiguous(),
            order.contiguous(),
            wt,
            bias.contiguous(),
            offsets.contiguous(),
            coeff.contiguous(),
            out,
            x.shape[0],
            n_rbf,
            n_out,
            start,
            spacing,
            half_band,
            BLOCK_P=BLOCK_P,
            BLOCK_C=FORWARD_BLOCK_C,
            BLOCK_K=BLOCK_K,
            DOT=x.dtype == torch.float32,
            num_warps=FORWARD_WARPS,
        )
    return out


def gaussian_rbf_linear_grad_triton(
    x: torch.Tensor,
    order: torch.Tensor,
    grad: torch.Tensor,
    offsets: torch.Tensor,
    coeff: torch.Tensor,
    start: float,
    spacing: float,
    half_band: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_out, n_rbf = grad.shape[1], offsets.shape[0]
    gwt = grad.new_zeros(n_rbf, n_out)
    gbias = grad.new_zeros(n_out)
    if x.shape[0] > 0 and n_out > 0:
        grid = (triton.cdiv(x.shape[0], BLOCK_P), triton.cdiv(n_out, BLOCK_C))
        wrap_triton(_grad_kernel)[grid](
            x.contiguous(),
            order.contiguous(),
            grad.contiguous(),
            offsets.contiguous(),
            coeff.contiguous(),
            gwt,
            gbias,
            x.shape[0],
            n_rbf,
            n_out,
            start,
            spacing,
            half_band,
            BLOCK_P=BLOCK_P,
            BLOCK_C=BLOCK_C,
            BLOCK_K=BLOCK_K,
            DOT=x.dtype == torch.float32,
        )
    return gwt, gbias

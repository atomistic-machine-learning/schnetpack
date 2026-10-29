"""Triton kernels for :mod:`schnetpack.nn.ops.message` (imported only when Triton is).

Both kernels tile pairs x features. ``message_sum`` gathers the neighbor rows, multiplies
by the pair weights and adds the products into the center rows with atomics, so the
per-pair product is never written to memory and the pair order does not matter. The
summation order, and with it the last bits of float32 results, varies between runs.
"""

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = ["message_sum_triton", "pair_product_triton", "supported"]


def supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Inputs the kernels handle; everything else takes the PyTorch path."""
    return (
        a.is_cuda
        and a.dtype == b.dtype
        and a.dtype in (torch.float32, torch.float64)
        and a.shape[1:] == b.shape[1:]
    )


def _blocks(n_features: int) -> tuple[int, int]:
    block_f = min(128, triton.next_power_of_2(max(n_features, 1)))
    return 4096 // block_f, block_f


@triton.jit
def _message_sum_kernel(
    x_ptr,
    w_ptr,
    idx_i_ptr,
    idx_j_ptr,
    out_ptr,
    n_pairs,
    n_features,
    stride_x,
    stride_w,
    stride_out,
    BLOCK_P: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask_p = p < n_pairs
    mask = mask_p[:, None] & (f < n_features)[None, :]
    i = tl.load(idx_i_ptr + p, mask=mask_p, other=0)
    j = tl.load(idx_j_ptr + p, mask=mask_p, other=0)
    x = tl.load(x_ptr + j[:, None] * stride_x + f[None, :], mask=mask, other=0.0)
    w = tl.load(
        w_ptr + p[:, None].to(tl.int64) * stride_w + f[None, :], mask=mask, other=0.0
    )
    tl.atomic_add(out_ptr + i[:, None] * stride_out + f[None, :], x * w, mask=mask)


@triton.jit
def _pair_product_kernel(
    a_ptr,
    b_ptr,
    idx_a_ptr,
    idx_b_ptr,
    out_ptr,
    n_pairs,
    n_features,
    stride_a,
    stride_b,
    stride_out,
    BLOCK_P: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask_p = p < n_pairs
    mask = mask_p[:, None] & (f < n_features)[None, :]
    ia = tl.load(idx_a_ptr + p, mask=mask_p, other=0)
    ib = tl.load(idx_b_ptr + p, mask=mask_p, other=0)
    a = tl.load(a_ptr + ia[:, None] * stride_a + f[None, :], mask=mask, other=0.0)
    b = tl.load(b_ptr + ib[:, None] * stride_b + f[None, :], mask=mask, other=0.0)
    out = out_ptr + p[:, None].to(tl.int64) * stride_out + f[None, :]
    tl.store(out, a * b, mask=mask)


def _rows(t: torch.Tensor) -> torch.Tensor:
    # 2d with contiguous rows; trailing feature dims are flattened
    t = t.reshape(t.shape[0], -1)
    return t if t.stride(1) == 1 else t.contiguous()


def message_sum_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    x2, w2 = _rows(x), _rows(w)
    n_pairs, n_features = w2.shape
    out = w2.new_zeros(dim_size, n_features)
    if n_pairs > 0 and n_features > 0:
        block_p, block_f = _blocks(n_features)
        grid = (triton.cdiv(n_pairs, block_p), triton.cdiv(n_features, block_f))
        wrap_triton(_message_sum_kernel)[grid](
            x2,
            w2,
            idx_i.contiguous(),
            idx_j.contiguous(),
            out,
            n_pairs,
            n_features,
            x2.stride(0),
            w2.stride(0),
            out.stride(0),
            BLOCK_P=block_p,
            BLOCK_F=block_f,
        )
    return out.view((dim_size,) + tuple(w.shape[1:]))


def pair_product_triton(
    a: torch.Tensor, b: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    a2, b2 = _rows(a), _rows(b)
    n_pairs, n_features = idx_a.shape[0], a2.shape[1]
    out = a2.new_empty(n_pairs, n_features)
    if n_pairs > 0 and n_features > 0:
        block_p, block_f = _blocks(n_features)
        grid = (triton.cdiv(n_pairs, block_p), triton.cdiv(n_features, block_f))
        wrap_triton(_pair_product_kernel)[grid](
            a2,
            b2,
            idx_a.contiguous(),
            idx_b.contiguous(),
            out,
            n_pairs,
            n_features,
            a2.stride(0),
            b2.stride(0),
            out.stride(0),
            BLOCK_P=block_p,
            BLOCK_F=block_f,
        )
    return out.view((n_pairs,) + tuple(a.shape[1:]))

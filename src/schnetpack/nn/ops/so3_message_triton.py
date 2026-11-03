"""Triton kernels for :mod:`schnetpack.nn.ops.so3_message` (imported only when Triton is).

One program per atom x tile of ``BLOCK_F`` features keeps its sum in registers and writes
each output row once; nothing of shape [pairs, features] is stored. The forward kernel
visits the pairs of a center atom, so they must be sorted by center, and the adjoint kernel
those of a neighbor, through a permutation, because that gradient sums on the neighbors.
Each row block is multiplied by the pair's coupling per l with a ``tl.dot`` in IEEE
precision (no TF32, whatever the matmul precision setting). Only the gradient of the
coupling goes through atomics: tiles of features share a pair.
"""

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = ["message_triton", "adjoint_triton", "supported"]

# 64 features with 4 warps took 0.30 ms for 28,160 pairs, 9 rows and 3 l on an H100 PCIe,
# against 0.32-0.41 ms for 32 or 128 features or 8 warps
BLOCK_F, WARPS = 64, 4


def supported(x: torch.Tensor) -> bool:
    """Inputs the kernels handle; everything else takes the PyTorch path."""
    return x.is_cuda and x.dtype in (torch.float32, torch.float64)


def _blocks(n_s: int, n_f: int) -> tuple[int, int]:
    """Tile sizes: powers of 2, and at least the 16 rows ``tl.dot`` needs."""
    block_s = max(16, triton.next_power_of_2(n_s))
    return block_s, max(16, min(BLOCK_F, triton.next_power_of_2(n_f)))


@triton.jit
def _atom_rows(ptr, atom, s, f, n_s, n_f, mask):
    """The rows of one atom, [BLOCK_S, BLOCK_F]."""
    return tl.load(
        ptr + (atom.to(tl.int64) * n_s + s)[:, None] * n_f + f[None, :],
        mask=mask,
        other=0.0,
    )


@triton.jit
def _coupling(g_ptr, p_l, s, n_s, mask):
    """The coupling of one pair and l, [BLOCK_S, BLOCK_S]: rows s, columns s2."""
    return tl.load(
        g_ptr + (p_l * n_s + s)[:, None] * n_s + s[None, :], mask=mask, other=0.0
    )


@triton.jit
def _message_kernel(
    x_ptr,
    w_ptr,
    g_ptr,
    idx_j_ptr,
    start_ptr,
    out_ptr,
    n_s,
    n_l,
    n_f,
    BLOCK_S: tl.constexpr,
    BLOCK_F: tl.constexpr,
    N_L: tl.constexpr,
):
    i = tl.program_id(0)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    s = tl.arange(0, BLOCK_S)
    mask_f = f < n_f
    mask_s = s < n_s
    mask = mask_s[:, None] & mask_f[None, :]
    mask_g = mask_s[:, None] & mask_s[None, :]
    acc = tl.zeros((BLOCK_S, BLOCK_F), dtype=x_ptr.dtype.element_ty)
    for p in range(tl.load(start_ptr + i), tl.load(start_ptr + i + 1)):
        xj = _atom_rows(x_ptr, tl.load(idx_j_ptr + p), s, f, n_s, n_f, mask)
        for l in tl.static_range(N_L):
            p_l = p.to(tl.int64) * n_l + l
            g = _coupling(g_ptr, p_l, s, n_s, mask_g)
            w = tl.load(w_ptr + p_l * n_f + f, mask=mask_f, other=0.0)
            acc += tl.dot(g, xj, input_precision="ieee") * w[None, :]
    tl.store(
        out_ptr + (i.to(tl.int64) * n_s + s)[:, None] * n_f + f[None, :], acc, mask=mask
    )


@triton.jit
def _adjoint_kernel(
    grad_ptr,
    x_ptr,
    w_ptr,
    g_ptr,
    idx_i_ptr,
    perm_ptr,
    start_j_ptr,
    dx_ptr,
    dw_ptr,
    dg_ptr,
    n_s,
    n_l,
    n_f,
    BLOCK_S: tl.constexpr,
    BLOCK_F: tl.constexpr,
    N_L: tl.constexpr,
    WANT_X: tl.constexpr,
    WANT_W: tl.constexpr,
    WANT_G: tl.constexpr,
):
    j = tl.program_id(0)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    s = tl.arange(0, BLOCK_S)
    mask_f = f < n_f
    mask_s = s < n_s
    mask = mask_s[:, None] & mask_f[None, :]
    mask_g = mask_s[:, None] & mask_s[None, :]
    if WANT_W or WANT_G:
        xj = _atom_rows(x_ptr, j, s, f, n_s, n_f, mask)
    if WANT_X:
        dx = tl.zeros((BLOCK_S, BLOCK_F), dtype=x_ptr.dtype.element_ty)
    for k in range(tl.load(start_j_ptr + j), tl.load(start_j_ptr + j + 1)):
        p = tl.load(perm_ptr + k)
        grad = _atom_rows(grad_ptr, tl.load(idx_i_ptr + p), s, f, n_s, n_f, mask)
        for l in tl.static_range(N_L):
            p_l = p.to(tl.int64) * n_l + l
            g = _coupling(g_ptr, p_l, s, n_s, mask_g)
            if WANT_X or WANT_G:
                w = tl.load(w_ptr + p_l * n_f + f, mask=mask_f, other=0.0)
            if WANT_X or WANT_W:
                h = tl.dot(tl.trans(g), grad, input_precision="ieee")  # G^T grad
            if WANT_X:
                dx += h * w[None, :]
            if WANT_W:
                tl.store(dw_ptr + p_l * n_f + f, tl.sum(h * xj, axis=0), mask=mask_f)
            if WANT_G:
                tl.atomic_add(
                    dg_ptr + (p_l * n_s + s)[:, None] * n_s + s[None, :],
                    tl.dot(grad * w[None, :], tl.trans(xj), input_precision="ieee"),
                    mask=mask_g,
                )
    if WANT_X:
        tl.store(
            dx_ptr + (j.to(tl.int64) * n_s + s)[:, None] * n_f + f[None, :],
            dx,
            mask=mask,
        )


def message_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    start: torch.Tensor,
    perm: torch.Tensor,
    start_j: torch.Tensor,
) -> torch.Tensor:
    """The forward kernel; ``idx_i``, ``perm``, ``start_j`` are for its gradient."""
    n_a, n_s, n_f = x.shape
    n_l = w.shape[1]
    out = x.new_empty(x.shape)
    if n_a > 0 and n_f > 0:
        block_s, block_f = _blocks(n_s, n_f)
        wrap_triton(_message_kernel)[(n_a, triton.cdiv(n_f, block_f))](
            x.contiguous(),
            w.contiguous(),
            g.contiguous(),
            idx_j.contiguous(),
            start.contiguous(),
            out,
            n_s,
            n_l,
            n_f,
            BLOCK_S=block_s,
            BLOCK_F=block_f,
            N_L=n_l,
            num_warps=WARPS,
        )
    return out


def adjoint_triton(
    grad: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    start: torch.Tensor,
    perm: torch.Tensor,
    start_j: torch.Tensor,
    want_x: bool,
    want_w: bool,
    want_g: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The adjoint kernel; ``idx_j`` and ``start`` are for its gradient."""
    n_a, n_s, n_f = x.shape
    n_l = w.shape[1]
    # the skipped pairs add nothing, hence zeros; an output not asked for stays empty
    dx = x.new_empty(x.shape if want_x else 0)
    dw = w.new_zeros(w.shape) if want_w else w.new_empty(0)
    dg = g.new_zeros(g.shape) if want_g else g.new_empty(0)
    if n_a > 0 and n_f > 0:
        block_s, block_f = _blocks(n_s, n_f)
        wrap_triton(_adjoint_kernel)[(n_a, triton.cdiv(n_f, block_f))](
            grad.contiguous(),
            x.contiguous(),
            w.contiguous(),
            g.contiguous(),
            idx_i.contiguous(),
            perm.contiguous(),
            start_j.contiguous(),
            dx,
            dw,
            dg,
            n_s,
            n_l,
            n_f,
            BLOCK_S=block_s,
            BLOCK_F=block_f,
            N_L=n_l,
            WANT_X=want_x,
            WANT_W=want_w,
            WANT_G=want_g,
            num_warps=WARPS,
        )
    return dx, dw, dg

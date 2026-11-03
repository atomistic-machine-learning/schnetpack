"""The message sum of :class:`~schnetpack.nn.SO3Convolution` as one op.

``so3_message(x, w, g, idx_i, idx_j, cutoff)`` computes::

    y[i, s, f] = sum_{pairs p of i} sum_{l, s2} w[p, l, f] g[p, l, s, s2] x[idx_j[p], s2, f]

with the filters ``w`` per l and the coupling ``g``, which is the Clebsch-Gordan tensor
contracted with the spherical harmonics of the pair and has no feature dimension. The
sparse sum in ``SO3Convolution`` runs over every Clebsch-Gordan term of every pair and
feature; here each neighbor's rows are multiplied by ``g`` at once, on the GPU, in a Triton
kernel (:mod:`schnetpack.nn.ops.so3_message_triton`).

SO3net with 128 features and 3 interactions, ms per training step on an H100 PCIe:

=====================================  ======  ======
                                       sparse  kernel
=====================================  ======  ======
QM9 energy, batch 100, lmax 2            9.52    4.97
QM9 energy, batch 100, lmax 3           32.75    7.36
aspirin E + F, batch 100, lmax 2        26.28   23.41
aspirin E + F, batch 100, lmax 3        95.20   31.42
aspirin E + F, batch 10, lmax 2          3.94    4.07
aspirin E + F, batch 10, lmax 3         11.42    4.91
=====================================  ======  ======

The gain follows the Clebsch-Gordan terms (10, 83 and 353 at lmax 1, 2 and 3). lmax 1 does
not gain: the kernel's 16 x 16 product is padded for 4 rows.

The sum is linear in each of ``grad``, ``x``, ``w`` and ``g``, for any ``grad`` on the
output::

    T(grad, x, w, g) = sum_{p, l, s, s2, f} grad[idx_i[p], s, f] w[p, l, f] g[p, l, s, s2]
                                            x[idx_j[p], s2, f]

Each derivative of ``T`` is ``T`` with one slot left open, so forces and a force loss
(double backward) run through the same two kernels: the forward one, open at ``grad``, and
the adjoint one, open at any of the others.
"""

import torch

__all__ = ["so3_message", "so3_message_pairs", "so3_message_supported"]

try:
    from schnetpack.nn.ops import so3_message_triton
except ImportError:  # no Triton: SO3Convolution takes its sparse sum on CUDA as well
    so3_message_triton = None


def _add(total: torch.Tensor | None, term: torch.Tensor) -> torch.Tensor:
    return term if total is None else total + term


if so3_message_triton is not None:
    _op = torch.library.triton_op(
        "schnetpack::so3_message", so3_message_triton.message_triton, mutates_args=()
    )
    _adjoint_op = torch.library.triton_op(
        "schnetpack::so3_message_adjoint",
        so3_message_triton.adjoint_triton,
        mutates_args=(),
    )

    def _setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def _backward(ctx, grad):
        x, w, g, *pairs = ctx.saved_tensors
        want_x, want_w, want_g = ctx.needs_input_grad[:3]
        dx, dw, dg = _adjoint_op(grad, x, w, g, *pairs, want_x, want_w, want_g)
        grads = (dx if want_x else None, dw if want_w else None, dg if want_g else None)
        return grads + (None,) * len(pairs)

    def _adjoint_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-3])
        ctx.want = inputs[-3:]

    def _adjoint_backward(ctx, ax, aw, ag):
        grad, x, w, g, *pairs = ctx.saved_tensors
        want_x, want_w, want_g = ctx.want
        need_grad, need_x, need_w, need_g = ctx.needs_input_grad[:4]
        # a cotangent fills the slot its output left open, and T stays open at the others
        terms = []
        if want_x:
            terms.append((ax, w, g, False, need_w, need_g))
        if want_w:
            terms.append((x, aw, g, need_x, False, need_g))
        if want_g:
            terms.append((x, w, ag, need_x, need_w, False))
        d_grad = d_x = d_w = d_g = None
        for tx, tw, tg, open_x, open_w, open_g in terms:
            if need_grad:
                d_grad = _add(d_grad, _op(tx, tw, tg, *pairs))
            if open_x or open_w or open_g:
                dx, dw, dg = _adjoint_op(
                    grad, tx, tw, tg, *pairs, open_x, open_w, open_g
                )
                d_x = _add(d_x, dx) if open_x else d_x
                d_w = _add(d_w, dw) if open_w else d_w
                d_g = _add(d_g, dg) if open_g else d_g
        return (d_grad, d_x, d_w, d_g) + (None,) * (len(pairs) + 3)

    _op.register_autograd(_backward, setup_context=_setup)
    _adjoint_op.register_autograd(_adjoint_backward, setup_context=_adjoint_setup)


@torch.jit.unused
def so3_message_supported(x: torch.Tensor) -> bool:
    """Whether :func:`so3_message` can run on these features."""
    return so3_message_triton is not None and so3_message_triton.supported(x)


@torch.jit.unused
def so3_message_pairs(
    idx_i: torch.Tensor, idx_j: torch.Tensor, cutoff: torch.Tensor, n_atoms: int
) -> list[torch.Tensor]:
    """
    The order in which the kernels visit the pairs, for :func:`so3_message`.

    A sort takes about 30 kernels, so a model computes this once for all its convolutions.

    Pairs must be sorted by center: a program of the kernel handles one atom. Pairs with
    a zero cutoff are left out, since their filters are zero, from the trailing ones on
    in the forward kernel. These are the padding pairs of a padded batch; all on one
    atom, they would hold up the whole kernel. The cutoff function must also have a zero
    derivative there, as every compact cutoff has.

    Args:
        idx_i: center atom of each pair, sorted
        idx_j: neighbor of each pair
        cutoff: cutoff function of each pair, any shape with ``n_pairs`` elements
        n_atoms: number of atoms

    Returns:
        ``idx_i``, ``idx_j``, the first pair of each center, the pairs by neighbor and
        the first of those of each neighbor, all int32
    """
    atoms = torch.arange(n_atoms + 1, device=idx_i.device, dtype=torch.int32)
    idx_i, idx_j = idx_i.to(torch.int32), idx_j.to(torch.int32)
    inside = cutoff.reshape(-1) != 0
    n_used = inside.numel() - inside.flip(0).int().argmax() if inside.numel() else 0
    start = torch.searchsorted(idx_i, atoms).clamp(max=n_used).to(torch.int32)
    neighbor = torch.where(inside, idx_j, n_atoms)
    perm = torch.argsort(neighbor, stable=True).to(torch.int32)
    start_j = torch.searchsorted(neighbor[perm], atoms).to(torch.int32)
    return [idx_i, idx_j, start, perm, start_j]


@torch.jit.unused
def so3_message(
    x: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    pairs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Sum the pair messages of an SO3 convolution on the center atoms (module docstring).

    Args:
        x: atom features, shape ``(n_atoms, n_s, n_features)``
        w: filters, shape ``(n_pairs, n_l, n_features)``
        g: coupling, shape ``(n_pairs, n_l, n_s, n_s)``
        pairs: :func:`so3_message_pairs`

    Returns:
        messages, shape ``(n_atoms, n_s, n_features)``
    """
    return _op(x, w, g, *pairs)

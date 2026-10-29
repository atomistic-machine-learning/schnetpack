"""Message passing over atom pairs as two custom ops.

``message_sum(x, w, idx_i, idx_j, dim_size)``
    ``out[i] = sum over pairs p with idx_i[p] == i of x[idx_j[p]] * w[p]``: gather the
    neighbor features, weight them per pair and sum them on the center atom (SchNet's
    continuous-filter convolution).
``pair_product(a, b, idx_a, idx_b)``
    ``out[p] = a[idx_a[p]] * b[idx_b[p]]``: per-pair product of two atom features.

The gradient of each op is written with the two ops again, so forces, a force loss
(double backward) and any higher derivative run through the same two kernels. As
registered ops they stay single nodes in ``make_fx`` / ``torch.compile`` graphs.

The Triton kernels (:mod:`schnetpack.nn.ops.message_triton`) are opt-in: set the
environment variable ``SCHNETPACK_KERNELS=1`` or call ``set_kernels_enabled(True)``.
Off (the default), and under TorchScript, both functions are plain PyTorch, which
``torch.compile`` fuses with the surrounding ops; on an H100 that beat the kernels in
compiled models, while the kernels were 1.2-1.5x faster in eager mode on large systems.
Compiled graphs keep the setting they were traced with.
"""

import os

import torch

__all__ = ["message_sum", "pair_product", "set_kernels_enabled", "kernels_enabled"]

_KERNELS_ENABLED = os.environ.get("SCHNETPACK_KERNELS", "0") == "1"


def set_kernels_enabled(enabled: bool) -> None:
    """Switch the GPU kernels on or off (off: plain PyTorch on every device)."""
    global _KERNELS_ENABLED
    _KERNELS_ENABLED = enabled


def kernels_enabled() -> bool:
    return _KERNELS_ENABLED


def _message_sum_torch(
    x: torch.Tensor,
    w: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    shape = list(w.shape)
    shape[0] = dim_size
    # in place on a fresh tensor: out-of-place index_add would clone the zeros first
    return w.new_zeros(shape).index_add_(0, idx_i, x[idx_j] * w)


def _pair_product_torch(
    a: torch.Tensor, b: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    return a[idx_a] * b[idx_b]


try:
    from schnetpack.nn.ops import message_triton
except ImportError:  # no Triton: PyTorch bodies on CUDA as well
    message_triton = None


def _message_sum_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    if (
        message_triton is not None
        and _KERNELS_ENABLED
        and message_triton.supported(x, w)
    ):
        return message_triton.message_sum_triton(x, w, idx_i, idx_j, dim_size)
    return _message_sum_torch(x, w, idx_i, idx_j, dim_size)


def _pair_product_impl(
    a: torch.Tensor, b: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    if (
        message_triton is not None
        and _KERNELS_ENABLED
        and message_triton.supported(a, b)
    ):
        return message_triton.pair_product_triton(a, b, idx_a, idx_b)
    return _pair_product_torch(a, b, idx_a, idx_b)


if message_triton is not None:
    # triton_op: compiled graphs launch the kernels from generated code; a custom_op
    # would go back through Python dispatch on every call (slower on small systems).
    # The kernel switch is read when a graph is compiled.
    _message_sum_op = torch.library.triton_op(
        "schnetpack::message_sum", _message_sum_impl, mutates_args=()
    )
    _pair_product_op = torch.library.triton_op(
        "schnetpack::pair_product", _pair_product_impl, mutates_args=()
    )
else:
    _message_sum_op = torch.library.custom_op(
        "schnetpack::message_sum", _message_sum_impl, mutates_args=()
    )
    _pair_product_op = torch.library.custom_op(
        "schnetpack::pair_product", _pair_product_impl, mutates_args=()
    )

    @_message_sum_op.register_fake
    def _(x, w, idx_i, idx_j, dim_size):
        return w.new_empty((dim_size,) + tuple(w.shape[1:]))

    @_pair_product_op.register_fake
    def _(a, b, idx_a, idx_b):
        return a.new_empty((idx_a.shape[0],) + tuple(a.shape[1:]))


def _message_sum_setup(ctx, inputs, output):
    x, w, idx_i, idx_j, dim_size = inputs
    ctx.save_for_backward(x, w, idx_i, idx_j)


def _message_sum_backward(ctx, grad):
    x, w, idx_i, idx_j = ctx.saved_tensors
    dx = dw = None
    if ctx.needs_input_grad[0]:
        dx = _message_sum_op(grad, w, idx_j, idx_i, x.shape[0])
    if ctx.needs_input_grad[1]:
        dw = _pair_product_op(grad, x, idx_i, idx_j)
    return dx, dw, None, None, None


def _pair_product_setup(ctx, inputs, output):
    a, b, idx_a, idx_b = inputs
    ctx.save_for_backward(a, b, idx_a, idx_b)


def _pair_product_backward(ctx, grad):
    a, b, idx_a, idx_b = ctx.saved_tensors
    da = db = None
    if ctx.needs_input_grad[0]:
        da = _message_sum_op(b, grad, idx_a, idx_b, a.shape[0])
    if ctx.needs_input_grad[1]:
        db = _message_sum_op(a, grad, idx_b, idx_a, b.shape[0])
    return da, db, None, None


_message_sum_op.register_autograd(
    _message_sum_backward, setup_context=_message_sum_setup
)
_pair_product_op.register_autograd(
    _pair_product_backward, setup_context=_pair_product_setup
)


@torch.jit.unused
def _message_sum_dispatch(
    x: torch.Tensor,
    w: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    if not _KERNELS_ENABLED:
        return _message_sum_torch(x, w, idx_i, idx_j, dim_size)
    return _message_sum_op(x, w, idx_i, idx_j, dim_size)


@torch.jit.unused
def _pair_product_dispatch(
    a: torch.Tensor, b: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    if not _KERNELS_ENABLED:
        return _pair_product_torch(a, b, idx_a, idx_b)
    return _pair_product_op(a, b, idx_a, idx_b)


def message_sum(
    x: torch.Tensor,
    w: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    Sum pair-weighted neighbor features on the center atoms.

    ``out[i] = sum over pairs p with idx_i[p] == i of x[idx_j[p]] * w[p]``, the same
    as ``scatter_add(x[idx_j] * w, idx_i, dim_size)`` without the per-pair tensor.

    Args:
        x: atom features of the neighbors, shape ``(n_atoms, ...)``
        w: pair weights, shape ``(n_pairs, ...)`` with the same trailing shape as x
        idx_i: index of the center atom of each pair
        idx_j: index of the neighbor of each pair
        dim_size: number of center atoms

    Returns:
        summed messages, shape ``(dim_size, ...)``
    """
    # TorchScript models (spkdeploy, LAMMPS) must not depend on the Python-registered op
    if torch.jit.is_scripting():
        return _message_sum_torch(x, w, idx_i, idx_j, dim_size)
    return _message_sum_dispatch(x, w, idx_i, idx_j, dim_size)


def pair_product(
    a: torch.Tensor, b: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    """
    Per-pair product of two atom features, ``out[p] = a[idx_a[p]] * b[idx_b[p]]``.

    Args:
        a: atom features, shape ``(n_atoms_a, ...)``
        b: atom features, shape ``(n_atoms_b, ...)`` with the same trailing shape as a
        idx_a: row of a for each pair
        idx_b: row of b for each pair

    Returns:
        pair features, shape ``(n_pairs, ...)``
    """
    if torch.jit.is_scripting():
        return _pair_product_torch(a, b, idx_a, idx_b)
    return _pair_product_dispatch(a, b, idx_a, idx_b)

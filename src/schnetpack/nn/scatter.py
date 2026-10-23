import torch

__all__ = ["scatter_add"]


def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Plain torch rather than a scripted function: TorchScript would hand concrete sizes
    to graph tracers (``make_fx``, ``torch.compile``) and bake them into the graph.
    Models are still scriptable as a whole.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    shape = list(x.shape)
    shape[dim] = dim_size
    # in place on a fresh tensor: out-of-place index_add would clone the zeros first
    return x.new_zeros(shape).index_add_(dim, idx_i, x)

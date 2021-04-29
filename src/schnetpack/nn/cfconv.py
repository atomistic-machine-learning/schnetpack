import torch
from torch import nn

__all__ = ["CFConv"]


def cfconv(
    x: torch.Tensor,
    Wij: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
) -> torch.Tensor:
    """
    Continuous-filter convolution.

    Args:
        x: input values
        Wij: filter
        idx_i: index of center atom i
        idx_j: index of neighbors j
        reduce: reduction method (sum, mean, ...)

    Returns:
        convolved inputs

    """
    x_j = x[idx_j]  # torch.gather(x, 0, idx_j[:, None])
    x_ij = x_j * Wij
    # y = segment_coo(x_ij, idx_i, reduce=reduce)
    tmp = torch.zeros(x.shape, dtype=x_ij.dtype, device=x_ij.device)
    y = tmp.index_add(0, idx_i, x_ij)
    return y


class CFConv(nn.Module):
    """
    Continuous-filter convolution.
    """

    reduce: str

    def __init__(self):
        """
        Args:
            reduce: reduction method (sum, mean, ...)
        """
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        Wij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ):
        """
        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            convolved inputs
        """
        return cfconv(x, Wij, idx_i, idx_j)

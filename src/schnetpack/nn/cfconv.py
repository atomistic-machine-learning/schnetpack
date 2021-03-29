import torch
from torch import nn
from torch_scatter import segment_coo

__all__ = ["CFConv"]


def cfconv(
    x: torch.Tensor,
    Wij: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    reduce: str = "sum",
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
    x_ij = x[idx_j] * Wij
    y = segment_coo(x_ij, idx_i, reduce=reduce)
    return y


class CFConv(nn.Module):
    """
    Continuous-filter convolution.
    """

    reduce: str

    def __init__(self, reduce="sum"):
        """
        Args:
            reduce: reduction method (sum, mean, ...)
        """
        super().__init__()
        self.reduce = reduce

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
        return cfconv(x, Wij, idx_i, idx_j, self.reduce)

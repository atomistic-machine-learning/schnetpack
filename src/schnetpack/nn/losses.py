import torch
import torch.nn as nn

__all__ = ["L2Squared"]


class L2Squared(nn.Module):
    """Computes the squared L2 norm"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sum(x**2)
        print(y)
        return y

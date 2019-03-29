import numpy as np
import torch
from torch import nn as nn


__all__ = ["CosineCutoff", "MollifierCutoff", "HardCutoff"]


class CosineCutoff(nn.Module):
    """
    Class wrapper for cosine cutoff function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Interatomic distances.

        Returns:
            torch.Tensor: Values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class MollifierCutoff(nn.Module):
    """
    Class wrapper for mollifier cutoff function.

    Args:
        cutoff (float, optional): Cutoff radius.
        eps (float, optional):

    """

    def __init__(self, cutoff=5.0, eps=1.e-7):
        super(MollifierCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("eps", torch.FloatTensor([eps]))

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Interatomic distances.

        Returns:
            torch.Tensor: Values of cutoff function.
        """
        mask = (distances + self.eps < self.cutoff).float()
        exponent = 1.0 - 1.0 / (1.0 - torch.pow(distances * mask / self.cutoff, 2))
        cutoffs = torch.exp(exponent)
        cutoffs = cutoffs * mask
        return cutoffs


class HardCutoff(nn.Module):
    """
    Class wrapper for hard cutoff function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(HardCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Interatomic distances.

        Returns:
            torch.Tensor: Values of cutoff function.
        """
        mask = (distances <= self.cutoff).float()
        return distances * mask

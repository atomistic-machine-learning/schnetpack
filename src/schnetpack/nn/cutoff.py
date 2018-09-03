import numpy as np
import torch
from torch import nn as nn

__all__ = [
    'CosineCutoff', 'MollifierCutoff'
]


def cosine_cutoff(distances, cutoff=5.0):
    """
    Compute the Behler type cosine cutoff for the distances and return it in the
    form of an array with the same shape as the original distances.

    Args:
        distances (torch.Tensor): Interatomic distances (Nbatch x Nat x Nneigh)
        cutoff (float): Cutoff value, all values beyond are set to 0

    Returns:
        torch.Tensor: Tensor holding values of the cutoff function (Nbatch x Nat x Nneigh)
    """
    # Compute values of cutoff function
    cutoffs = 0.5 * (torch.cos(distances * np.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    cutoffs *= (distances < cutoff).float()
    # Add a singleton dimension for easier broadcasting
    cutoffs = torch.unsqueeze(cutoffs, -1)
    return cutoffs


class CosineCutoff(nn.Module):
    """
    Class wrapper for cosine cutoff function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer('cutoff', torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Interatomic distances.

        Returns:
            torch.Tensor: Values of cutoff function.
        """
        return cosine_cutoff(distances, cutoff=self.cutoff)


def mollifier_cutoff(distances, cutoff=5.0):
    """
    Infinitely differentiable cutoff based on a mollifier function.
    (See https://en.wikipedia.org/wiki/Mollifier)

    Args:
        distances (torch.Tensor): Interatomic distances (Nbatch x Nat x Nneigh)
        cutoff (float): Cutoff value, all values beyond are set to 0

    Returns:
        torch.Tensor: Tensor holding values of the cutoff function (Nbatch x Nat x Nneigh)
    """
    mask = (distances <= cutoff).float()
    exponent = 1.0 - 1.0 / (1.0 - torch.pow(distances * mask / cutoff, 2))
    cutoffs = torch.exp(exponent)
    cutoffs = cutoffs * mask
    cutoffs = torch.unsqueeze(cutoffs, -1)
    return cutoffs


class MollifierCutoff(nn.Module):
    """
    Class wrapper for mollifier cutoff function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(MollifierCutoff, self).__init__()
        self.register_buffer('cutoff', torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Interatomic distances.

        Returns:
            torch.Tensor: Values of cutoff function.
        """
        return mollifier_cutoff(distances, cutoff=self.cutoff)

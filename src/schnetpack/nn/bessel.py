from math import pi
import torch
import torch.nn as nn

__all__ = ["BesselBasis"]

class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        a = self.freqs[None, None, None, :]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm

        return y


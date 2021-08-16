from math import pi

import torch
import torch.nn as nn
import numpy as np
import schnetpack.nn as snn
__all__ = ["gaussian_rbf", "GaussianRBF", "GaussianRBFCentered", "RBF_PhysNet"]

from torch import nn as nn


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class GaussianRBFCentered(nn.Module):
    r"""Gaussian radial basis functions centered at the origin."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 1.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: width of last Gaussian function, :math:`\mu_{N_g}`
            start: width of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBFCentered, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        widths = torch.linspace(start, cutoff, n_rbf)
        offset = torch.zeros_like(widths)
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

<<<<<<< HEAD
    
def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBF_PhysNet(nn.Module):
    
    def __init__(self, n_rbf, cutoff = 10.0):
        
        super(RBF_PhysNet,self).__init__()
        
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = softplus_inverse(torch.linspace(1.0,np.exp(-cutoff),n_rbf))
        centers = snn.activations.shifted_softplus(((centers)))
        
        widths = [softplus_inverse((0.5/((1.0-np.exp(-cutoff))/n_rbf))**2)]*n_rbf
        widths = snn.activations.shifted_softplus(torch.Tensor(widths))
        
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        
        
    def forward(self, r_ij):
        r_ij = r_ij.unsqueeze(-1)
        g_ij = torch.exp(-self.widths*(torch.exp(-r_ij)-self.centers)**2)
        
        return g_ij
=======

class BesselRBF(nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).

    References:

    .. [#dimenet] Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf

        freqs = torch.arange(1, n_rbf + 1) * pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y
>>>>>>> kts/outmods

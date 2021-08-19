import math
import torch
from torch import nn


__all__ = ["CosineCutoff", "MollifierCutoff", "mollifier_cutoff", "cosine_cutoff", "PhysNetCutOff"]


def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.

        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}

        Args:
            cutoff (float, optional): cutoff radius.

        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class PhysNetCutOff(nn.Module):
    """Cutoff Function from Physnet
    
       References:
        .. [#Cutoff] Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S. Texturing & Modeling: A Procedural Approach;
        Morgan Kaufmann, 2003
        .. [#Cutoff] O.Unke, M.Meuwly
        PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges
        https://arxiv.org/abs/1902.08408
    """
    
    def __init__(self, cutoff: float):
        super(PhysNetCutOff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, d_ij: torch.Tensor):


        # Compute values of cutoff function
        input_cut = 1 - 6*(d_ij/self.cutoff)**5 + 15*(d_ij/self.cutoff)**4 - 10*(d_ij/self.cutoff)**3
        # Remove contributions beyond the cutoff radius
        input_cut *= (d_ij < self.cutoff).float()
        return input_cut


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)


def mollifier_cutoff(input: torch.Tensor, cutoff: torch.Tensor, eps: torch.Tensor):
    r""" Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.

    """
    mask = (input + eps < cutoff).float()
    exponent = 1.0 - 1.0 / (1.0 - torch.pow(input * mask / cutoff, 2))
    cutoffs = torch.exp(exponent)
    cutoffs = cutoffs * mask
    return cutoffs


class MollifierCutoff(nn.Module):
    r""" Mollifier cutoff module scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    """

    def __init__(self, cutoff: float, eps: float = 1.0e-7):
        """
        Args:
            cutoff: Cutoff radius.
            eps: Offset added to distances for numerical stability.
        """
        super(MollifierCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("eps", torch.FloatTensor([eps]))

    def forward(self, input: torch.Tensor):
        return mollifier_cutoff(input, self.cutoff, self.eps)

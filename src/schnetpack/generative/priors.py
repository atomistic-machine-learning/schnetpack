"""
Starting distributions for sampling.

A prior is a distribution, not geometry — which is why it lives here rather than
on the path. :class:`PathPrior` reads a path's terminal noise level and hands
back the matching Gaussian; deliberately different starting distributions
(harmonic, covariance-matched, scaffold) plug in through the same interface
without a path saying anything about them. Starting below t_max from a
structured prior pairs with
:meth:`~schnetpack.generative.sampler.Sampler.denoise`.
"""

import abc
from typing import Optional, Sequence

import torch

from schnetpack.generative.paths import Path

__all__ = ["Prior", "PathPrior"]


class Prior(abc.ABC):
    """Distribution over starting states for the reverse process."""

    @abc.abstractmethod
    def sample(
        self,
        shape: Sequence[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class PathPrior(Prior):
    """
    The terminal distribution N(0, sigma(t_max)^2 I) of a path.

    Exact when alpha(t_max) = 0 (flow matching); for the diffusion paths it is
    the usual approximation that the residual alpha(t_max) x0 term is
    negligible, i.e. that the data has been drowned out by t_max.
    """

    def __init__(self, path: Path):
        """
        Args:
            path: path whose terminal distribution to sample
        """
        self.path = path

    def sample(self, shape, dtype=None, device=None):
        t_max = torch.full((1,), self.path.t_max, dtype=dtype, device=device)
        sigma_max = self.path.sigma(t_max).item()
        return sigma_max * torch.randn(*shape, dtype=dtype, device=device)

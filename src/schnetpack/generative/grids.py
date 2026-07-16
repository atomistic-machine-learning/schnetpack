"""
Time grids — where along the path a sampler places its steps.

Separating the grid from the solver is the other half of EDM's decoupling: how
many steps you take, and where, is independent of what each step computes. A
uniform grid wastes steps at high noise, where the reverse process barely
moves, and starves the low-noise end, where the detail appears; a warped grid
just moves them without touching the integrator.
"""

import abc
from typing import Optional

import torch

__all__ = ["TimeGrid", "UniformGrid", "KarrasGrid"]


class TimeGrid(abc.ABC):
    """Builds the sequence of times a sampler steps through."""

    @abc.abstractmethod
    def __call__(
        self,
        t_start: float,
        t_end: float,
        n_steps: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Args:
            t_start: first time (the noisy end when denoising)
            t_end: last time (the clean end when denoising)
            n_steps: number of integrator steps

        Returns:
            Monotone grid of shape (n_steps + 1,), decreasing when denoising.
        """
        raise NotImplementedError


class UniformGrid(TimeGrid):
    """Evenly spaced times — the obvious default, and right for VP-type paths."""

    def __call__(self, t_start, t_end, n_steps, dtype=None, device=None):
        return torch.linspace(t_start, t_end, n_steps + 1, dtype=dtype, device=device)


class KarrasGrid(TimeGrid):
    """
    The rho-warped grid of Karras et al. (2022):

        t_i = ( t_start^(1/rho) + i/n (t_end^(1/rho) - t_start^(1/rho)) )^rho

    rho > 1 packs steps toward ``t_end``, where the truncation error of a step
    is largest. rho = 7 is the paper's empirical optimum; rho = 1 recovers
    :class:`UniformGrid`.

    Warps t directly, so it belongs to paths whose time *is* the noise level —
    :class:`~schnetpack.generative.paths.EDMPath` above all. On a VP path,
    where t and sigma are related nonlinearly, the warp has no such meaning.

    Note the grid stops at ``t_end`` rather than appending a final t = 0 as the
    paper does: paths carry a nonzero ``t_min`` for good reasons, and the step
    that would take you to zero noise is just a denoiser evaluation there —
    ``parametrization.to_x0(model(x, t_min), x, t_min, path)`` — which callers
    who want it can do in one line.
    """

    def __init__(self, rho: float = 7.0):
        """
        Args:
            rho: warp exponent; higher packs more steps near t_end
        """
        self.rho = rho

    def __call__(self, t_start, t_end, n_steps, dtype=None, device=None):
        i = torch.arange(n_steps + 1, dtype=dtype, device=device)
        inv = 1.0 / self.rho
        warped = t_start**inv + (i / n_steps) * (t_end**inv - t_start**inv)
        return warped**self.rho

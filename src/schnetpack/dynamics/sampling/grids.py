"""
Time grids — where along the path a sampler places its steps.

Separating the grid from the solver means how many steps you take, and where,
is independent of what each step computes. A uniform grid wastes steps at
high noise, where the reverse process barely moves, and starves the low-noise
end, where the detail appears; a warped grid just moves them without touching
the integrator.
"""

import abc
from typing import Optional

import torch

__all__ = ["TimeGrid", "UniformGrid"]


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

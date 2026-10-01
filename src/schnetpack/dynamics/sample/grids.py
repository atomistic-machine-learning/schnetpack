"""
Time grids: where along the path a sampler places its steps, independently
of what each step computes.
"""

import abc

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
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
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
    """Evenly spaced times."""

    def __call__(self, t_start, t_end, n_steps, dtype=None, device=None):
        return torch.linspace(t_start, t_end, n_steps + 1, dtype=dtype, device=device)

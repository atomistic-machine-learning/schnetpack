"""
Time grids: where along the path a sampler places its steps, independently
of what each step computes. A grid owns the time range it covers.
"""

import abc

import torch

__all__ = ["TimeGrid", "UniformGrid"]


class TimeGrid(abc.ABC):
    """Builds the sequence of times a sampler steps through, t_start -> t_end."""

    def __init__(self, t_start: float = 1.0, t_end: float = 0.0):
        """
        Args:
            t_start: first time (the noisy end when denoising); below the
                prior's end for partial denoising
            t_end: last time (the clean end when denoising)
        """
        self.t_start = t_start
        self.t_end = t_end

    @abc.abstractmethod
    def __call__(
        self,
        n_steps: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Args:
            n_steps: number of integrator steps

        Returns:
            Monotone grid of shape (n_steps + 1,) from t_start to t_end,
            decreasing when denoising.
        """
        raise NotImplementedError


class UniformGrid(TimeGrid):
    """Evenly spaced times."""

    def __call__(self, n_steps, dtype=None, device=None):
        return torch.linspace(
            self.t_start, self.t_end, n_steps + 1, dtype=dtype, device=device
        )

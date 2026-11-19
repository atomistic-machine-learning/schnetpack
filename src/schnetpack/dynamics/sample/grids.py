"""
Time grids: where along the path a sampler places its steps, independently
of what each step computes. A grid owns the time range it covers.
"""

import abc

import torch

__all__ = ["TimeGrid", "UniformGrid", "KarrasGrid"]


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


class KarrasGrid(TimeGrid):
    """Times evenly spaced in t^(1/rho) from t_start down to t_min, ``n_steps`` of
    them, then t_end: the steps get shorter towards the clean end. This is the
    noise-level grid of Karras et al. (2022, eq. 5) written in path time; for a
    schedule whose noise scale is proportional to t (e.g. ``VELinear``) it is the
    same grid."""

    def __init__(
        self,
        t_min: float,
        rho: float = 7.0,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ):
        """
        Args:
            t_min: last time before the final step to t_end
            rho: the larger, the more steps near t_min
            t_start: first time
            t_end: last time
        """
        super().__init__(t_start, t_end)
        self.t_min = t_min
        self.rho = rho

    def __call__(self, n_steps, dtype=None, device=None):
        ramp = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float64)
        start, low = self.t_start ** (1 / self.rho), self.t_min ** (1 / self.rho)
        t = (start + ramp * (low - start)) ** self.rho
        t = torch.cat([t, t.new_full((1,), self.t_end)])
        return t.to(dtype=dtype, device=device)

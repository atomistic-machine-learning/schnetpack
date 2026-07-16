"""
Numerical solvers for (reverse-time) diffusion processes.

An integrator consumes only ``process.drift`` and ``process.diffusion`` — it is
agnostic to what it integrates: a reverse SDE, a probability-flow ODE (g = 0)
or a forward process. Denoising runs backwards in time, so ``dt < 0``.
"""

import abc

import torch

__all__ = ["Integrator"]


class Integrator(abc.ABC):
    """Base class for SDE/ODE solvers."""

    @abc.abstractmethod
    def step(
        self, process, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Advance x from t to t + dt.

        Args:
            process: object exposing drift(x, t) and diffusion(t)
            x: current state, shape (batch, ...)
            t: current time, shape (batch,)
            dt: time increment (0-dim tensor; negative when denoising)
        """
        raise NotImplementedError

    def integrate(self, process, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Run steps along a monotone time grid.

        Args:
            process: object exposing drift(x, t) and diffusion(t)
            x: initial state, shape (batch, ...)
            ts: time grid of shape (n_steps + 1,), decreasing for denoising,
                e.g. ``torch.linspace(T, t_min, n_steps + 1)``
        """
        for i in range(ts.shape[0] - 1):
            t = ts[i].expand(x.shape[0])
            dt = ts[i + 1] - ts[i]
            x = self.step(process, x, t, dt)
        return x

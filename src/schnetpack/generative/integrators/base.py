"""
Numerical solvers for (reverse-time) diffusion processes.

An integrator consumes only ``dynamics.drift`` and ``dynamics.diffusion`` — it
is agnostic to what it integrates: a reverse SDE, a probability-flow ODE (g = 0)
or a forward process. Denoising runs backwards in time, so ``dt < 0``.
"""

import abc

import torch

__all__ = ["Integrator"]


class Integrator(abc.ABC):
    """Base class for SDE/ODE solvers."""

    requires_sde: bool = False
    """Whether this integrator steps through the (f, g) chart's closed forms.

    False for the generic solvers, which consume only ``drift`` and
    ``diffusion`` and work on any reverse process. True for the ones that
    discretize through the chart itself — the ancestral steps, which read
    the exact posterior or the raw score — so the
    :class:`~schnetpack.generative.sampler.Sampler` can demand a
    :class:`~schnetpack.generative.differential_equations.ReverseSDE` at assembly instead of
    failing mid-run.
    """

    @abc.abstractmethod
    def step(
        self, dynamics, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Advance x from t to t + dt.

        Args:
            dynamics: object exposing drift(x, t) and diffusion(t)
            x: current state, shape (batch, ...)
            t: current time, shape (batch,)
            dt: time increment (0-dim tensor; negative when denoising)
        """
        raise NotImplementedError

    def integrate(self, dynamics, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Run steps along a monotone time grid.

        Args:
            dynamics: object exposing drift(x, t) and diffusion(t)
            x: initial state, shape (batch, ...)
            ts: time grid of shape (n_steps + 1,), decreasing for denoising,
                e.g. ``torch.linspace(T, t_min, n_steps + 1)``
        """
        for i in range(ts.shape[0] - 1):
            t = ts[i].expand(x.shape[0])
            dt = ts[i + 1] - ts[i]
            x = self.step(dynamics, x, t, dt)
        return x

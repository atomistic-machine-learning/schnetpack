"""
Numerical solvers for (reverse-time) diffusion processes.

An integrator consumes ``dynamics.drift`` and ``dynamics.diffusion`` and is
agnostic to what it integrates. Denoising runs backwards in time, so
``dt < 0``. Catalog: ``docs_new/sampling.md`` §2.
"""

import abc

import torch

__all__ = ["Integrator"]


class Integrator(abc.ABC):
    """Base class for SDE/ODE solvers."""

    requires_sde: bool = False
    """Whether this integrator steps through the (f, g) chart's closed forms.

    True for the ancestral steps, which read the exact posterior or the raw
    score; the :class:`~schnetpack.dynamics.sampling.sampler.Sampler` then
    demands a :class:`~schnetpack.generative.differential_equations.ReverseSDE`
    at assembly.
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

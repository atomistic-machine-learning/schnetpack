"""
Integrators: the step rule a sampler applies along its time grid.

An integrator reads the reverse process through its sampler — the drift
``sampler.drift(batch, x, t)`` and the diffusion ``sampler.diffusion(t)``,
None for a deterministic one — so :class:`Euler` and :class:`Heun` serve the
:class:`~schnetpack.dynamics.sample.sampler.SDESampler` and the
:class:`~schnetpack.dynamics.sample.sampler.ODESampler` alike. Ancestral
sampling is not an integrator but its own
:class:`~schnetpack.dynamics.sample.sampler.AncestralSampler`.
"""

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from schnetpack.generative.noising import expand_t

if TYPE_CHECKING:
    from schnetpack.dynamics.sample.sampler import Sampler

__all__ = ["Integrator", "Euler", "Heun"]


class Integrator(abc.ABC):
    """Advances a sampler's state by one step of its time grid."""

    @abc.abstractmethod
    def step(
        self,
        sampler: "Sampler",
        batch: Mapping[str, Any],
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance x from t to t + dt.

        Args:
            sampler: provides the drift and diffusion of the reverse process
            batch: the current batch, carried along to every field evaluation
            x: current state, shape (n_rows, ...)
            t: current time, shape (n_rows,)
            dt: time increment (0-dim tensor; negative when denoising)

        Returns:
            The advanced state.
        """
        raise NotImplementedError


class Euler(Integrator):
    """
    First-order step: x <- x + drift dt + g sqrt(|dt|) z (Euler–Maruyama);
    plain Euler without diffusion. One field evaluation per step.
    """

    def step(self, sampler, batch, x, t, dt):
        x_new = x + sampler.drift(batch, x, t) * dt
        g = sampler.diffusion(t)
        if g is None:
            return x_new
        return x_new + expand_t(g, x) * dt.abs().sqrt() * torch.randn_like(x)


class Heun(Integrator):
    """
    Second-order Heun step on the drift plus an Euler–Maruyama diffusion
    increment. Without diffusion this is the EDM (Karras et al. 2022)
    sampler. Two field evaluations per step.
    """

    def step(self, sampler, batch, x, t, dt):
        f1 = sampler.drift(batch, x, t)
        f2 = sampler.drift(batch, x + f1 * dt, t + dt)
        x_new = x + 0.5 * (f1 + f2) * dt
        g = sampler.diffusion(t)
        if g is None:
            return x_new
        return x_new + expand_t(g, x) * dt.abs().sqrt() * torch.randn_like(x)

"""Euler–Maruyama solver (plain Euler for deterministic processes)."""

import torch

from schnetpack.generative.integrators.base import Integrator
from schnetpack.generative.processes import expand_t

__all__ = ["EulerMaruyama"]


class EulerMaruyama(Integrator):
    """
    First-order solver: x <- x + f dt + g sqrt(|dt|) z.

    For g = 0 (probability-flow ODE) the noise term vanishes and this is the
    plain Euler method.
    """

    def step(self, dynamics, x, t, dt):
        x = x + dynamics.drift(x, t) * dt
        g = expand_t(dynamics.diffusion(t), x)
        return x + g * dt.abs().sqrt() * torch.randn_like(x)

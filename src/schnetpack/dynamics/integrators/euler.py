"""Euler–Maruyama solver (plain Euler for deterministic processes)."""

import torch

from schnetpack.dynamics.integrators.base import Integrator
from schnetpack.generative.processes import expand_t

__all__ = ["EulerMaruyama"]


class EulerMaruyama(Integrator):
    """First-order solver: x <- x + f dt + g sqrt(|dt|) z; plain Euler when g = 0."""

    def step(self, dynamics, x, t, dt):
        x = x + dynamics.drift(x, t) * dt
        g = expand_t(dynamics.diffusion(t), x)
        return x + g * dt.abs().sqrt() * torch.randn_like(x)

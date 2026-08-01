"""Heun (second-order) solver."""

import torch

from schnetpack.generative.integrators.base import Integrator
from schnetpack.generative.processes import expand_t

__all__ = ["Heun"]


class Heun(Integrator):
    """
    Second-order Heun step on the drift; the diffusion contribution is added
    as an Euler–Maruyama increment.

    On the probability-flow ODE (eta = 0) this is the deterministic
    second-order sampler popularized by EDM (Karras et al. 2022), which reaches
    comparable sample quality with far fewer function evaluations than
    first-order solvers.
    """

    def step(self, dynamics, x, t, dt):
        f1 = dynamics.drift(x, t)
        x_pred = x + f1 * dt
        f2 = dynamics.drift(x_pred, t + dt)
        x = x + 0.5 * (f1 + f2) * dt
        g = expand_t(dynamics.diffusion(t), x)
        return x + g * dt.abs().sqrt() * torch.randn_like(x)

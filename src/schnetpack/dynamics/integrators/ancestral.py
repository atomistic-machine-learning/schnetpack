"""Ancestral sampling as integrators."""

import torch

from schnetpack.dynamics.integrators.base import Integrator
from schnetpack.generative.processes import expand_t

__all__ = ["Ancestral", "AncestralDDPM"]


class Ancestral(Integrator):
    """
    Exact-posterior ancestral step: estimate x0 from the model's score
    (:meth:`~schnetpack.generative.differential_equations.SDE.x0_from_score`),
    then draw x_s ~ p(x_s | x_t, x0_hat)
    (:meth:`~schnetpack.generative.differential_equations.SDE.posterior`).

    One class for every process with a Gaussian kernel: the textbook DDPM
    step on VP, the NCSN/GPFF ancestral sampler on VE. Intrinsically
    stochastic, so it ignores the reverse process's ``churn``.
    """

    requires_sde = True

    def step(self, dynamics, x, t, dt):
        x0_hat = dynamics.sde.x0_from_score(x, dynamics.score(x, t), t)
        mean, std = dynamics.sde.posterior(x, x0_hat, t, t + dt)
        return mean + expand_t(std, x) * torch.randn_like(x)


class AncestralDDPM(Integrator):
    """
    DDPM ancestral step in score form, with beta_k = g(t)^2 |dt|:

        x_{k-1} = (x_k + beta_k * score) / sqrt(1 - beta_k) + sqrt(beta_k) z.

    A discretization of the reverse VP process using the DDPM
    ``sigma_t^2 = beta_t`` variance; needs a VP-type process. Intrinsically
    stochastic, so it ignores the reverse process's ``churn``.
    """

    requires_sde = True

    def step(self, dynamics, x, t, dt):
        beta = expand_t(dynamics.g2(t), x) * dt.abs()
        mean = (x + beta * dynamics.score(x, t)) / torch.sqrt(1.0 - beta)
        return mean + beta.sqrt() * torch.randn_like(x)

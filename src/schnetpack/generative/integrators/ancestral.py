"""Ancestral sampling as integrators."""

import torch

from schnetpack.generative.integrators.base import Integrator
from schnetpack.generative.processes import expand_t

__all__ = ["Ancestral", "AncestralDDPM"]


class Ancestral(Integrator):
    """
    Generic ancestral update — the exact posterior step

        x0_hat ~ model,   x_s ~ p(x_s | x_t, x0_hat),

    i.e. estimate x0 from the model, then draw from the closed-form Gaussian
    posterior of the chart
    (:meth:`~schnetpack.generative.differential_equations.SDE.posterior`). Schedule
    logic lives entirely in that closed form, so one class covers every
    process with a Gaussian kernel: on VP it is the textbook DDPM ancestral
    step with the exact (beta-tilde) posterior variance, on VE it reduces to
    the familiar score-form update x + score (sigma_t^2 - sigma_s^2) plus
    matched noise — the GPFF/NCSN ancestral sampler.

    Like :class:`AncestralDDPM` it discretizes the reverse process through
    something other than drift/diffusion — the reverse process's ``score``,
    converted to an x0-estimate through the chart
    (:meth:`~schnetpack.generative.differential_equations.SDE.x0_from_score`),
    then stepped through the chart's posterior — and, being intrinsically
    stochastic, it ignores the reverse process's ``churn``. ``requires_sde``
    is how it says so: the Sampler then assembles a ReverseSDE even at
    churn = 0, and a configuration without the Gaussian kernel is refused
    at assembly.
    """

    requires_sde = True

    def step(self, dynamics, x, t, dt):
        x0_hat = dynamics.sde.x0_from_score(x, dynamics.score(x, t), t)
        mean, std = dynamics.sde.posterior(x, x0_hat, t, t + dt)
        return mean + expand_t(std, x) * torch.randn_like(x)


class AncestralDDPM(Integrator):
    """
    DDPM ancestral update — the exact discrete-time posterior step

        x_{k-1} = (x_k + beta_k * score) / sqrt(1 - beta_k) + sqrt(beta_k) z,
        beta_k  = g(t)^2 |dt|,

    i.e. a particular discretization of the reverse VP process. Unlike the
    generic solvers it is written in terms of the raw score rather than the
    drift, so it needs a
    :class:`~schnetpack.generative.differential_equations.ReverseSDE` (for its ``g2``
    and ``score``) built on a VP-type path — hence ``requires_sde``. That is
    a deliberate exception to the rule that integrators see only drift and
    diffusion: the step *is* a statement about the score, and rewriting it
    through the drift would only obscure it.

    Uses the DDPM ``sigma_t^2 = beta_t`` variance choice. Being intrinsically
    stochastic, it ignores the reverse process's ``churn``: a Sampler
    configured with ``churn=0`` and this integrator still samples the SDE.
    """

    requires_sde = True

    def step(self, dynamics, x, t, dt):
        beta = expand_t(dynamics.g2(t), x) * dt.abs()
        mean = (x + beta * dynamics.score(x, t)) / torch.sqrt(1.0 - beta)
        return mean + beta.sqrt() * torch.randn_like(x)

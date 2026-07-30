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
    posterior the process already knows
    (:meth:`~schnetpack.generative.processes.Process.posterior`). Schedule
    logic lives entirely in that closed form, so one class covers every
    process with a Gaussian kernel: on VP it is the textbook DDPM ancestral
    step with the exact (beta-tilde) posterior variance, on VE it reduces to
    the familiar score-form update x + score (sigma_t^2 - sigma_s^2) plus
    matched noise — the GPFF/NCSN ancestral sampler.

    Like :class:`AncestralDDPM` it discretizes the reverse process through
    something other than drift/diffusion — here the
    :class:`~schnetpack.generative.reverse.ReverseProcess`'s ``x0`` and the
    forward process's posterior — and, being intrinsically stochastic, it
    ignores the reverse process's ``churn``. Requires
    :attr:`~schnetpack.generative.processes.Process.has_gaussian_kernel`
    (the posterior raises otherwise).
    """

    def step(self, process, x, t, dt):
        x0_hat = process.x0(x, t)
        mean, std = process.process.posterior(x, x0_hat, t, t + dt)
        return mean + expand_t(std, x) * torch.randn_like(x)


class AncestralDDPM(Integrator):
    """
    DDPM ancestral update — the exact discrete-time posterior step

        x_{k-1} = (x_k + beta_k * score) / sqrt(1 - beta_k) + sqrt(beta_k) z,
        beta_k  = g(t)^2 |dt|,

    i.e. a particular discretization of the reverse VP process. Unlike the
    generic solvers it is written in terms of the raw score rather than the
    drift, so it needs a
    :class:`~schnetpack.generative.reverse.ReverseProcess` (for its ``g2``
    and ``score``) built on a VP-type path. That is a deliberate exception to
    the rule that integrators see only drift and diffusion: the step *is* a
    statement about the score, and rewriting it through the drift would only
    obscure it.

    Uses the DDPM ``sigma_t^2 = beta_t`` variance choice. Being intrinsically
    stochastic, it ignores the reverse process's ``churn``: a Sampler
    configured with ``churn=0`` and this integrator still samples the SDE.
    """

    def step(self, process, x, t, dt):
        beta = expand_t(process.g2(t), x) * dt.abs()
        mean = (x + beta * process.score(x, t)) / torch.sqrt(1.0 - beta)
        return mean + beta.sqrt() * torch.randn_like(x)

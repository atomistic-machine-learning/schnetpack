"""Exact DDPM ancestral sampling as an integrator."""

import torch

from schnetpack.generative.integrators.base import Integrator
from schnetpack.generative.paths import expand_t

__all__ = ["AncestralDDPM"]


class AncestralDDPM(Integrator):
    """
    DDPM ancestral update — the exact discrete-time posterior step

        x_{k-1} = (x_k + beta_k * score) / sqrt(1 - beta_k) + sqrt(beta_k) z,
        beta_k  = g(t)^2 |dt|,

    i.e. a particular discretization of the reverse VP process. Unlike the
    generic solvers it is written in terms of the raw score rather than the
    drift, so it needs a
    :class:`~schnetpack.generative.reverse.ReverseProcess` (for its ``path``
    and ``score``) built on a VP-type path. That is a deliberate exception to
    the rule that integrators see only drift and diffusion: the step *is* a
    statement about the score, and rewriting it through the drift would only
    obscure it.

    Uses the DDPM ``sigma_t^2 = beta_t`` variance choice. Being intrinsically
    stochastic, it ignores the reverse process's ``churn``: a Sampler
    configured with ``churn=0`` and this integrator still samples the SDE.
    """

    def step(self, process, x, t, dt):
        beta = expand_t(process.path.g2(t), x) * dt.abs()
        mean = (x + beta * process.score(x, t)) / torch.sqrt(1.0 - beta)
        return mean + beta.sqrt() * torch.randn_like(x)

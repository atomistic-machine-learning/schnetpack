"""
Parametrizations — what the network predicts, and everything that follows.

A parametrization is bound to a :class:`~schnetpack.generative.paths.Path` and
owns both directions of the contract with a generative head:

- :meth:`Parametrization.target` builds the training target from an endpoint
  pair (consumed by :mod:`schnetpack.generative.losses`);
- the ``to_*`` methods convert a raw output into the canonical fields
  (consumed by reverse processes at sampling time);
- :meth:`Parametrization.reverse` builds the reverse-time process.

The field math lives here rather than on the path because a target is the
*definition* of a parametrization, not a property of a noise schedule. Adding a
parametrization must not mean editing ``paths.py`` — that would be the
parametrization axis reaching into the path axis, which is exactly what the
separation exists to prevent. The path supplies alpha, sigma and their
derivatives; this module decides what to do with them.

The binding runs this way round because the dependency does: every method here
needs a path, while a path is perfectly usable without a parametrization (for
noising, priors, or just its schedule). So the parametrization holds the path,
never the reverse.

Two identities are load-bearing and worth stating up front.

1. ``f sigma^2 - sigma sigma' = -1/2 g^2`` — immediate from the definition of
   g^2. Hence

       velocity = f x - 1/2 g^2 score

   is *exactly* the probability-flow ODE drift, which is why :meth:`to_velocity`
   and the PF-ODE never need separate code paths.

2. The Anderson (1982) reverse-time drift f x - 1/2 (1 + eta^2) g^2 s equals
   v - 1/2 eta^2 g^2 s. So the reverse process is a one-knob family around the
   velocity, with ``churn = eta^2`` (see
   :mod:`schnetpack.generative.reverse`). At churn = 0 a velocity-predicting
   model is used directly, and the singular velocity -> score conversion never
   runs.

Everything routes through the score: each ``to_score`` is a parametrization's
way *in*, and the base :meth:`to_velocity` / :meth:`to_x0` are the shared way
*out*. A parametrization that *is* one of the fields overrides that field to
return the output untouched — which matters, because the generic route back can
be singular exactly where the direct one is exact.

The model itself stays a bare callable ``model(x, t, cond) -> raw output``.
Nothing here wraps it.
"""

import abc
from typing import TYPE_CHECKING, Optional

import torch

from schnetpack.generative.paths import Path, expand_t

if TYPE_CHECKING:
    from schnetpack.generative.reverse import ReverseProcess

__all__ = [
    "Parametrization",
    "ScoreParametrization",
    "EpsParametrization",
    "X0Parametrization",
    "VelocityParametrization",
    "PseudoForceParametrization",
]


class Parametrization(abc.ABC):
    """Contract between a raw model output and the score/velocity/x0 fields."""

    def __init__(self, path: Path):
        """
        Args:
            path: interpolant supplying the schedule this parametrization reads
        """
        self.path = path

    # -- training --------------------------------------------------------- #

    @abc.abstractmethod
    def target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training target the head regresses.

        Takes the endpoint pair rather than the diffused sample on purpose:
        recovering x1 from x_t means dividing by sigma, which is zero at t = 0.
        The caller drew x1, so handing it over is free — and it leaves every
        target here a multiply-add, bar the score's.

        Args:
            x0: data endpoint
            x1: prior endpoint (the noise, under the independent coupling)
            t: path time, per-sample or scalar
            eps: bridge noise realization; unused while gamma is zero, and
                reserved for the bridge targets that will need it
        """
        raise NotImplementedError

    # -- sampling --------------------------------------------------------- #

    @abc.abstractmethod
    def to_score(
        self, output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Score of the marginal p_t, from the raw output."""
        raise NotImplementedError

    def to_velocity(
        self, output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Probability-flow velocity v = f x - 1/2 g^2 score, from the raw output."""
        score = self.to_score(output, x_t, t)
        f = expand_t(self.path.f(t), x_t)
        g2 = expand_t(self.path.g2(t), x_t)
        return f * x_t - 0.5 * g2 * score

    def to_x0(
        self, output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoised sample x0 = (x + sigma^2 score) / alpha, from the raw output.

        Singular where alpha -> 0 (flow matching at t = 1, VP at large t).
        """
        score = self.to_score(output, x_t, t)
        alpha = expand_t(self.path.alpha(t), x_t)
        sigma = expand_t(self.path.sigma(t), x_t)
        return (x_t + sigma**2 * score) / alpha

    # -- reverse ---------------------------------------------------------- #

    def reverse(self, model, churn: float = 1.0, cond=None) -> "ReverseProcess":
        """
        Reverse-time process driven by ``model``, as a
        :class:`~schnetpack.generative.reverse.ReverseProcess`.

        Args:
            model: callable (x, t, cond) -> raw output in this parametrization
            churn: stochasticity in [0, 1]; 1 = reverse-time SDE, 0 =
                probability-flow ODE. Equals eta^2 of the Anderson family.
            cond: conditioning passed through to the model
        """
        from schnetpack.generative.reverse import ReverseProcess

        return ReverseProcess(self, model, churn=churn, cond=cond)

    def probability_flow(self, model, cond=None) -> "ReverseProcess":
        """Deterministic ODE sharing the path's marginals."""
        return self.reverse(model, churn=0.0, cond=cond)


class ScoreParametrization(Parametrization):
    """
    The head predicts the score directly.

    Its target -x1 / sigma is the only one that divides, so it grows without
    bound as sigma -> 0 and spans whatever range sigma does. On a geometric VE
    schedule that is orders of magnitude, and an unweighted L2 will see only the
    low-noise end — pass ``weight=lambda t: path.sigma(t)**2`` to the loss,
    which makes the objective identical to noise matching.
    """

    def target(self, x0, x1, t, eps=None):
        return -x1 / expand_t(self.path.sigma(t), x1)

    def to_score(self, output, x_t, t):
        return output


class EpsParametrization(Parametrization):
    """
    The head predicts the noise (DDPM convention); the target is x1 itself.

    Valid only for Gaussian x1, where x1 *is* the noise realization — under a
    bridge coupling x1 is a data endpoint and this target is meaningless.
    """

    def target(self, x0, x1, t, eps=None):
        return x1

    def to_score(self, output, x_t, t):
        return -output / expand_t(self.path.sigma(t), output)


class X0Parametrization(Parametrization):
    """
    The head predicts the clean sample — the denoiser convention EDM builds on.

    Pairs with :class:`~schnetpack.generative.preconditioning.PrecondDenoiser`,
    which turns a raw net into a preconditioned denoiser without changing this
    contract.
    """

    def target(self, x0, x1, t, eps=None):
        return x0

    def to_score(self, output, x_t, t):
        # Tweedie: score = (alpha x0_hat - x) / sigma^2
        alpha = expand_t(self.path.alpha(t), x_t)
        sigma = expand_t(self.path.sigma(t), x_t)
        return (alpha * output - x_t) / sigma**2

    def to_x0(self, output, x_t, t):
        # Direct: the round trip through the score would divide by alpha.
        return output


class VelocityParametrization(Parametrization):
    """
    The head predicts the velocity d/dt x_t — the flow-matching convention.

    Its target alpha' x0 + sigma' x1 is the only one valid for every coupling,
    which is why flow, OT and bridge matching all regress it.

    Note that :meth:`to_score` inverts a relation that degenerates as g^2 -> 0
    (t -> 0 for the VE-type and flow-matching paths). Reverse processes only ask
    for the score when churn > 0, and their grids stop at ``path.t_min``; at
    churn = 0 the velocity is used directly and the inverse never runs.
    """

    def target(self, x0, x1, t, eps=None):
        alpha_dot = expand_t(self.path.alpha_dot(t), x0)
        sigma_dot = expand_t(self.path.sigma_dot(t), x1)
        return alpha_dot * x0 + sigma_dot * x1

    def to_score(self, output, x_t, t):
        # Invert v = f x - 1/2 g^2 s.
        f = expand_t(self.path.f(t), x_t)
        g2 = expand_t(self.path.g2(t), x_t)
        return 2.0 * (f * x_t - output) / g2

    def to_velocity(self, output, x_t, t):
        return output


class PseudoForceParametrization(Parametrization):
    """
    The head predicts a pseudo force F = 2 (x0 - x_t) — the GPFF convention.

    F is the negative gradient of the pseudo-energy ||x_t - x0||^2, so the head
    answers "which way, and how far, back to a clean sample". Substituting the
    interpolant gives the target without ever forming x_t::

        F = 2 (x0 - (alpha x0 + sigma x1)) = 2 ((1 - alpha) x0 - sigma x1)

    It is x0 up to an affine map, so it is exact wherever
    :class:`X0Parametrization` is and shares its best property: recovering x0
    costs no division (x0 = x_t + F/2), so nothing degenerates as sigma -> 0.

    What makes it worth a class of its own is what happens on a *variance
    exploding* path, where alpha = 1 and the target collapses to

        F = -2 sigma x1

    — the noise, scaled by how far the sample was pushed. The scale of F then
    carries sigma, so a sampler can estimate the noise level from the prediction
    alone and the head needs no time input at all. That is the whole point of the
    method, and it is VE-specific: on a VP-type path the (1 - alpha) x0 term
    mixes the data back in and the magnitude no longer reads as sigma.

    The same scaling is the cost. An eps head regresses a unit-variance target at
    every noise level; this one regresses a target whose scale runs with sigma,
    over the orders of magnitude a geometric VE schedule spans. Unweighted, the
    large-sigma end is the only thing an L2 can see. Pass

        weight=lambda t: (1.0 / path.sigma(t) ** 2).clamp(max=1.0)

    to the loss. The 1/sigma^2 undoes the scaling exactly — it makes the
    objective noise matching again — and the clip is what keeps it distinct from
    an eps head: it stops a handful of nearly-clean samples, where the unclipped
    weight would reach 1/sigma_min^2, from dominating every gradient, at the
    price of spending capacity where the correction is large rather than where it
    is small.
    """

    def target(self, x0, x1, t, eps=None):
        alpha = expand_t(self.path.alpha(t), x0)
        sigma = expand_t(self.path.sigma(t), x1)
        return 2.0 * ((1.0 - alpha) * x0 - sigma * x1)

    def to_x0(self, output, x_t, t):
        # Direct, and exact at sigma = 0 — the definition of F rearranged.
        return x_t + 0.5 * output

    def to_score(self, output, x_t, t):
        # Tweedie on the recovered x0; for VE this reduces to F / (2 sigma^2).
        alpha = expand_t(self.path.alpha(t), x_t)
        sigma = expand_t(self.path.sigma(t), x_t)
        return (alpha * self.to_x0(output, x_t, t) - x_t) / sigma**2

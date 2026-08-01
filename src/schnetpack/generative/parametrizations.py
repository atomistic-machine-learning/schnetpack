"""
Parametrizations — what the network predicts, and everything that follows.

A parametrization is *stateless*: pure field math, holding nothing. Every
method takes the
:class:`~schnetpack.generative.processes.Process` it is applied
to, and the consumers that need both — :class:`~schnetpack.generative.transforms.Diffuse`,
:class:`~schnetpack.generative.losses.MatchingLoss`,
:class:`~schnetpack.generative.sampler.Sampler`, the reverse processes
(:class:`~schnetpack.generative.sde.ReverseSDE`,
:class:`~schnetpack.generative.reverse.ReverseODE`) — take the pair
``(process, parametrization)`` explicitly. It owns both directions of the
contract with a generative head:

- :meth:`Parametrization.target` builds the training target from an endpoint
  pair (consumed by :mod:`schnetpack.generative.losses`);
- the ``to_*`` methods convert a raw output into the canonical fields
  (consumed by reverse processes at sampling time).

The field math lives here rather than on the process because a target is the
*definition* of a parametrization, not a property of a noise schedule. Adding
a parametrization must not mean editing ``processes.py`` — that would be the
parametrization axis reaching into the schedule axis, which is exactly what
the separation exists to prevent. The process is the one interface everything
here reads: the dimensionless a, b and their derivatives, the endpoint's
scale ``process.std``, and the noise level sigma(t) = b(t) * prior.std —
declared once, on the prior, and never mirrored. Nothing here touches
``process.prior`` directly.

Validity still settles at assembly, not mid-run: the score and noise targets
are statements about a Gaussian kernel, so :class:`ScoreParametrization` and
:class:`EpsParametrization` override :meth:`Parametrization.validate` to
demand :attr:`~schnetpack.generative.processes.Process.has_gaussian_kernel`
— and every consumer calls ``parametrization.validate(process)`` in its
constructor. The x0, velocity and pseudo-force targets are plain conditional
expectations and accept any process. What the split gives up is a single
bound object: training and sampling each name the pair, so keeping them
consistent (same process on both sides) is the caller's job — share the
objects, don't rebuild them.

Two identities are load-bearing and worth stating up front.

1. ``f b^2 - b b' = -1/2 g^2`` — immediate from the definition of
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
from typing import Optional

import torch

from schnetpack.generative.processes import Process, expand_t

__all__ = [
    "Parametrization",
    "ScoreParametrization",
    "EpsParametrization",
    "X0Parametrization",
    "VelocityParametrization",
    "PseudoForceParametrization",
]


def _require_gaussian(process: Process, cls: type) -> None:
    """Raise unless ``process`` has the one-sided Gaussian kernel."""
    obstruction = process.gaussian_kernel_obstruction()
    if obstruction is not None:
        raise TypeError(
            f"{cls.__name__} regresses a target that is a statement about a "
            f"Gaussian kernel, which this process does not have: "
            f"{obstruction}. Fix the configuration, or switch to a "
            "velocity, x0 or pseudo-force parametrization — those targets "
            "are plain conditional expectations, valid for any process."
        )


class Parametrization(abc.ABC):
    """
    Contract between a raw model output and the score/velocity/x0 fields.

    Stateless: every method takes the process it is applied to. Consumers
    call :meth:`validate` at their own construction, so an invalid pairing
    still fails at assembly.
    """

    velocity_needs_chart: bool = True
    """Whether :meth:`to_velocity` crosses the (f, g) chart.

    True on the base: the generic route to the velocity goes through the
    score and the probability-flow identity v = f x - 1/2 g^2 s, both chart
    statements. A parametrization whose head *is* the velocity overrides
    this to False — its conversion returns the output untouched — which is
    what lets :func:`~schnetpack.generative.reverse.reverse` assemble a
    chart-free :class:`~schnetpack.generative.reverse.ReverseODE` for it at
    churn = 0, the one reverse route valid for any endpoint law.
    """

    def validate(self, process: Process) -> None:
        """
        Raise unless this parametrization's target is meaningful for
        ``process``. Called by every consumer constructor.
        """

    # -- training --------------------------------------------------------- #

    @abc.abstractmethod
    def target(
        self,
        process: Process,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training target the head regresses.

        Takes the endpoint pair rather than the diffused sample on purpose:
        recovering x1 from x_t means dividing by b, which is zero at t = 0.
        The process drew x1, so handing it over is free — and it leaves every
        target here a multiply-add, bar the score's.

        Args:
            process: forward process the pair was drawn from; supplies the
                path geometry and the endpoint scale
            x0: data endpoint
            x1: prior endpoint (the noise, up to scale, under a Gaussian
                prior)
            t: path time, per-sample or scalar
            eps: bridge noise realization, exactly the one
                :meth:`~schnetpack.generative.processes.Process.perturb`
                drew; unused while gamma is zero, and reserved for the bridge
                targets that will need it
        """
        raise NotImplementedError

    # -- sampling --------------------------------------------------------- #

    @abc.abstractmethod
    def to_score(
        self,
        process: Process,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Score of the marginal p_t, from the raw output."""
        raise NotImplementedError

    def to_velocity(
        self,
        process: Process,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Probability-flow velocity v = f x - 1/2 g^2 score, from the raw output."""
        score = self.to_score(process, output, x_t, t)
        sde = process.sde()
        f = expand_t(sde.f(t), x_t)
        g2 = expand_t(sde.g2(t), x_t)
        return f * x_t - 0.5 * g2 * score

    def to_x0(
        self,
        process: Process,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoised sample x0 = (x + sigma^2 score) / a, from the raw output.

        Singular where a -> 0 (flow matching at t = 1, VP at large t).
        """
        score = self.to_score(process, output, x_t, t)
        a = expand_t(process.a(t), x_t)
        sigma = expand_t(process.sigma(t), x_t)
        return (x_t + sigma**2 * score) / a


class ScoreParametrization(Parametrization):
    """
    The head predicts the score directly.

    Requires a process with
    :attr:`~schnetpack.generative.processes.Process.has_gaussian_kernel`:
    the target below is the score *of the Gaussian kernel*, meaningless for
    any other endpoint.

    Its target -x1 / (b std^2) is the only one that divides, so it grows
    without bound as b -> 0 and spans whatever range b does. On a geometric
    VE schedule that is orders of magnitude, and an unweighted L2 will see
    only the low-noise end — pass ``weight=lambda t: path.b(t)**2`` to the
    loss, which makes the objective identical to noise matching up to the
    constant endpoint scale.
    """

    def validate(self, process):
        _require_gaussian(process, type(self))

    def target(self, process, x0, x1, t, eps=None):
        # score of p(x_t | x0): -(x_t - a x0) / sigma^2 = -x1 / (b std^2)
        return -x1 / expand_t(process.b(t) * process.std**2, x1)

    def to_score(self, process, output, x_t, t):
        return output


class EpsParametrization(Parametrization):
    """
    The head predicts the unit noise (DDPM convention): eps = x1 / std.

    The target keeps unit variance at every noise level and endpoint scale,
    which is the convention's whole appeal. Requires a process with
    :attr:`~schnetpack.generative.processes.Process.has_gaussian_kernel`,
    where x1 *is* a noise realization — under any other endpoint this target
    is meaningless.
    """

    def validate(self, process):
        _require_gaussian(process, type(self))

    def target(self, process, x0, x1, t, eps=None):
        return x1 / process.std

    def to_score(self, process, output, x_t, t):
        # score = -eps / sigma
        return -output / expand_t(process.sigma(t), output)


class X0Parametrization(Parametrization):
    """The head predicts the clean sample — the denoiser convention."""

    def target(self, process, x0, x1, t, eps=None):
        return x0

    def to_score(self, process, output, x_t, t):
        # Tweedie: score = (a x0_hat - x) / sigma^2
        a = expand_t(process.a(t), x_t)
        sigma = expand_t(process.sigma(t), x_t)
        return (a * output - x_t) / sigma**2

    def to_x0(self, process, output, x_t, t):
        # Direct: the round trip through the score would divide by a.
        return output


class VelocityParametrization(Parametrization):
    """
    The head predicts the velocity d/dt x_t — the flow-matching convention.

    Its target a' x0 + b' x1 is a plain conditional expectation, valid for
    every process — which is why flow, OT and bridge matching all regress it.

    Note that :meth:`to_score` inverts a relation that degenerates as g^2 -> 0
    (t -> 0 for the VE-type and flow-matching paths). Reverse processes only ask
    for the score when churn > 0, and their grids stop at ``path.t_min``; at
    churn = 0 the velocity is used directly and the inverse never runs.
    """

    velocity_needs_chart = False  # the head *is* the velocity

    def target(self, process, x0, x1, t, eps=None):
        a_dot = expand_t(process.a_dot(t), x0)
        b_dot = expand_t(process.b_dot(t), x1)
        return a_dot * x0 + b_dot * x1

    def to_score(self, process, output, x_t, t):
        # Invert v = f x - 1/2 g^2 s.
        sde = process.sde()
        f = expand_t(sde.f(t), x_t)
        g2 = expand_t(sde.g2(t), x_t)
        return 2.0 * (f * x_t - output) / g2

    def to_velocity(self, process, output, x_t, t):
        return output


class PseudoForceParametrization(Parametrization):
    """
    The head predicts a pseudo force F = 2 (x0 - x_t) — the GPFF convention.

    F is the negative gradient of the pseudo-energy ||x_t - x0||^2, so the head
    answers "which way, and how far, back to a clean sample". Substituting the
    interpolant gives the target without ever forming x_t::

        F = 2 (x0 - (a x0 + b x1)) = 2 ((1 - a) x0 - b x1)

    A plain conditional expectation like the x0 and velocity targets, so it
    is valid for *any* process — plain GPFF on a Gaussian-endpoint
    :class:`~schnetpack.generative.processes.VE`, GPFF with a shape prior or
    aligned noise on the same class with those parts swapped in. It is x0 up
    to an affine map, so it is exact wherever
    :class:`X0Parametrization` is and shares its best property: recovering x0
    costs no division (x0 = x_t + F/2), so nothing degenerates as b -> 0.

    What makes it worth a class of its own is what happens on a *variance
    exploding* path, where a = 1 and the target collapses to

        F = -2 b x1

    — the noise endpoint, scaled by how far the sample was pushed. The
    magnitude of F then carries the noise level sigma = b std, so a
    sampler can estimate it from the prediction alone and the head needs no
    time input at all. That is the whole point of the method, and it is
    VE-specific: on a VP-type path the (1 - a) x0 term mixes the data back in
    and the magnitude no longer reads as sigma.

    The same scaling is the cost. An eps head regresses a unit-variance target at
    every noise level; this one regresses a target whose scale runs with b,
    over the orders of magnitude a geometric VE schedule spans. Unweighted, the
    large-b end is the only thing an L2 can see. Pass

        weight=lambda t: (1.0 / path.b(t) ** 2).clamp(max=1.0)

    to the loss. The 1/b^2 undoes the scaling exactly — it makes the
    objective noise matching again — and the clip is what keeps it distinct
    from an eps head: it stops a handful of nearly-clean samples, where the
    unclipped weight would reach 1/b_min^2, from dominating every gradient,
    at the price of spending capacity where the correction is large rather
    than where it is small.
    """

    def target(self, process, x0, x1, t, eps=None):
        a = expand_t(process.a(t), x0)
        b = expand_t(process.b(t), x1)
        return 2.0 * ((1.0 - a) * x0 - b * x1)

    def to_x0(self, process, output, x_t, t):
        # Direct, and exact at b = 0 — the definition of F rearranged.
        # Needs no sigma either, which is what keeps GPFF's direct-denoising
        # sampler available under priors that declare no scalar scale.
        return x_t + 0.5 * output

    def to_score(self, process, output, x_t, t):
        # Tweedie on the recovered x0; for VE this reduces to F / (2 sigma^2).
        a = expand_t(process.a(t), x_t)
        sigma = expand_t(process.sigma(t), x_t)
        return (a * self.to_x0(process, output, x_t, t) - x_t) / sigma**2

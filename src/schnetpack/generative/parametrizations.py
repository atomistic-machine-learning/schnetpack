"""
Parametrizations: what the network predicts, and the conversions that follow.

A parametrization is stateless field math; every method takes the
:class:`~schnetpack.generative.processes.Process` it applies to. It owns the
training target (:meth:`Parametrization.target`) and the conversions of a raw
output into the canonical fields (``to_score``, ``to_velocity``, ``to_x0``).
Consumers that pair it with a process call :meth:`Parametrization.validate`
at construction. The model stays a bare callable ``model(x, t, cond)``.
Theory and the catalog: ``docs_new/parametrizations.md``.
"""

import abc

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

    Every conversion routes through the score by default; a parametrization
    that *is* one of the fields overrides that field to return the output
    untouched, since the generic route can be singular where the direct one
    is exact.
    """

    velocity_needs_chart: bool = True
    """Whether :meth:`to_velocity` crosses the (f, g) chart.

    False only for a head that predicts the velocity itself; the
    :class:`~schnetpack.dynamics.sampling.sampler.Sampler` then uses the
    chart-free :class:`~schnetpack.generative.differential_equations.ReverseODE`
    at churn = 0.
    """

    def validate(self, process: Process) -> None:  # noqa: B027 - optional hook
        """Raise unless this parametrization's target is meaningful for ``process``."""

    # -- training --------------------------------------------------------- #

    @abc.abstractmethod
    def target(
        self,
        process: Process,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training target the head regresses, from the endpoint pair.

        Args:
            process: forward process the pair was drawn from
            x0: data endpoint
            x1: prior endpoint
            t: path time, per-sample or scalar
            eps: bridge noise drawn by
                :meth:`~schnetpack.generative.processes.Process.perturb`;
                unused while gamma is zero
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
        """Denoised sample x0 = (x + sigma^2 score) / a; singular where a -> 0."""
        score = self.to_score(process, output, x_t, t)
        a = expand_t(process.a(t), x_t)
        sigma = expand_t(process.sigma(t), x_t)
        return (x_t + sigma**2 * score) / a


class ScoreParametrization(Parametrization):
    """
    The head predicts the score directly. Requires the Gaussian kernel.

    The target -x1 / (b std^2) spans whatever range b does; on a VE schedule
    pass ``weight=lambda t: process.b(t)**2`` to the loss.
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
    The head predicts the unit noise eps = x1 / std (DDPM convention).
    Requires the Gaussian kernel; the target has unit variance at every t.
    """

    def validate(self, process):
        _require_gaussian(process, type(self))

    def target(self, process, x0, x1, t, eps=None):
        return x1 / process.std

    def to_score(self, process, output, x_t, t):
        # score = -eps / sigma
        return -output / expand_t(process.sigma(t), output)


class X0Parametrization(Parametrization):
    """The head predicts the clean sample x0 (denoiser convention). Any process."""

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
    The head predicts the velocity d/dt x_t = a' x0 + b' x1 (flow matching).
    Any process.

    :meth:`to_score` divides by g^2, which vanishes as t -> 0 on VE and flow
    matching paths; it only runs at churn > 0, and the grids stop at t_min.
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
    The head predicts the pseudo force F = 2 (x0 - x_t) (GPFF convention).
    Any process.

    Recovering x0 = x_t + F/2 needs neither division nor sigma, which is what
    :class:`~schnetpack.dynamics.relax.DirectDenoising` relies on. On a VE
    path the target is -2 b x1, so |F| carries the noise level and the head
    needs no time input; the target's scale then runs with b, so pass
    ``weight=lambda t: (1 / process.b(t)**2).clamp(max=1.0)`` to the loss.
    """

    def target(self, process, x0, x1, t, eps=None):
        a = expand_t(process.a(t), x0)
        b = expand_t(process.b(t), x1)
        return 2.0 * ((1.0 - a) * x0 - b * x1)

    def to_x0(self, process, output, x_t, t):
        # Direct, and exact at b = 0 — the definition of F rearranged.
        return x_t + 0.5 * output

    def to_score(self, process, output, x_t, t):
        # Tweedie on the recovered x0; for VE this reduces to F / (2 sigma^2).
        a = expand_t(process.a(t), x_t)
        sigma = expand_t(process.sigma(t), x_t)
        return (a * self.to_x0(process, output, x_t, t) - x_t) / sigma**2

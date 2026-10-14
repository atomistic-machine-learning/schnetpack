"""
Parametrizations: what the network predicts, and the conversions that follow.

A parametrization is stateless field math; every method takes the
:class:`~schnetpack.generative.schedules.Schedule` it applies to. It owns the
training target (:meth:`Parametrization.target`) and the conversions of a raw
output into the canonical fields (``to_score``, ``to_velocity``, ``to_x0``).
Consumers that pair it with a schedule call :meth:`Parametrization.validate`
at construction. The model stays a bare callable ``model(x, t, cond)``.
Theory and the catalog: ``docs_new/parametrizations.md``.
"""

import abc
from collections.abc import Callable

import torch
from torch import nn

from schnetpack import properties
from schnetpack.generative.differential_equations import SDE
from schnetpack.generative.noising import expand_t
from schnetpack.generative.schedules import GaussianSchedule, Schedule

__all__ = [
    "Parametrization",
    "ScoreParametrization",
    "EpsParametrization",
    "X0Parametrization",
    "VelocityParametrization",
    "PseudoForceParametrization",
    "ParametrizedModel",
]


def _require_gaussian(schedule: Schedule, cls: type) -> None:
    """Raise unless ``schedule`` is a Gaussian schedule."""
    if not isinstance(schedule, GaussianSchedule):
        raise TypeError(
            f"{cls.__name__} regresses a target that is a statement about a "
            f"Gaussian kernel, which needs a GaussianSchedule; got "
            f"{type(schedule).__name__}. Switch to a velocity, x0 or "
            "pseudo-force parametrization — those targets are plain "
            "conditional expectations, valid for any schedule."
        )


class Parametrization(abc.ABC):
    """
    Contract between a raw model output and the score, noise, velocity, x0
    and pseudo-force fields.

    Every conversion routes through the score by default; a parametrization
    that *is* one of the fields overrides that field to return the output
    untouched, since the generic route can be singular where the direct one
    is exact.
    """

    velocity_needs_chart: bool = True
    """Whether :meth:`to_velocity` crosses the (f, g) chart.

    False only for a head that predicts the velocity itself; the
    :class:`~schnetpack.dynamics.sample.Sampler` then uses the
    chart-free :class:`~schnetpack.generative.differential_equations.ReverseODE`
    at eta2 = 0.
    """

    def validate(self, schedule: Schedule) -> None:  # noqa: B027 - optional hook
        """Raise unless this parametrization's target is meaningful for ``schedule``."""

    # -- training --------------------------------------------------------- #

    @abc.abstractmethod
    def target(
        self,
        schedule: Schedule,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training target the head regresses, from the endpoint pair.

        Args:
            schedule: schedule the pair was placed on
            x0: data endpoint
            x1: prior endpoint
            t: path time, per-sample or scalar
        """
        raise NotImplementedError

    # -- sampling --------------------------------------------------------- #

    @abc.abstractmethod
    def to_score(
        self,
        schedule: Schedule,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Score of the marginal p_t, from the raw output."""
        raise NotImplementedError

    def to_velocity(
        self,
        schedule: Schedule,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Probability-flow velocity v = f x - 1/2 g^2 score, from the raw output."""
        score = self.to_score(schedule, output, x_t, t)
        sde = SDE(schedule)
        f = expand_t(sde.f(t), x_t)
        g2 = expand_t(sde.g2(t), x_t)
        return f * x_t - 0.5 * g2 * score

    def to_x0(
        self,
        schedule: Schedule,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoised sample x0 = (x + sigma^2 score) / a; singular where a -> 0."""
        score = self.to_score(schedule, output, x_t, t)
        a = expand_t(schedule.a(t), x_t)
        sigma = expand_t(schedule.sigma(t), x_t)
        return (x_t + sigma**2 * score) / a

    def to_noise(
        self,
        schedule: Schedule,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Expected unit noise eps = -sigma score, from the raw output."""
        score = self.to_score(schedule, output, x_t, t)
        return -expand_t(schedule.sigma(t), x_t) * score

    def to_pseudo_force(
        self,
        schedule: Schedule,
        output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Pseudo force F = 2 (x0 - x_t), from the raw output."""
        return 2.0 * (self.to_x0(schedule, output, x_t, t) - x_t)


class ScoreParametrization(Parametrization):
    """
    The head predicts the score directly. Requires the Gaussian kernel.

    The target -x1 / (b std^2) spans whatever range b does; on a VE schedule
    pass ``weight=lambda t: schedule.b(t)**2`` to the loss.
    """

    def validate(self, schedule):
        _require_gaussian(schedule, type(self))

    def target(self, schedule, x0, x1, t):
        # score of p(x_t | x0): -(x_t - a x0) / sigma^2 = -x1 / (b std^2)
        return -x1 / expand_t(schedule.b(t) * schedule.sigma_max**2, x1)

    def to_score(self, schedule, output, x_t, t):
        return output


class EpsParametrization(Parametrization):
    """
    The head predicts the unit noise eps = x1 / std (DDPM convention).
    Requires the Gaussian kernel; the target has unit variance at every t.
    """

    def validate(self, schedule):
        _require_gaussian(schedule, type(self))

    def target(self, schedule, x0, x1, t):
        return x1 / schedule.sigma_max

    def to_score(self, schedule, output, x_t, t):
        # score = -eps / sigma
        return -output / expand_t(schedule.sigma(t), output)

    def to_noise(self, schedule, output, x_t, t):
        return output


class X0Parametrization(Parametrization):
    """The head predicts the clean sample x0 (denoiser convention). Any schedule."""

    def target(self, schedule, x0, x1, t):
        return x0

    def to_score(self, schedule, output, x_t, t):
        # Tweedie: score = (a x0_hat - x) / sigma^2
        a = expand_t(schedule.a(t), x_t)
        sigma = expand_t(schedule.sigma(t), x_t)
        return (a * output - x_t) / sigma**2

    def to_x0(self, schedule, output, x_t, t):
        # Direct: the round trip through the score would divide by a.
        return output


class VelocityParametrization(Parametrization):
    """
    The head predicts the velocity d/dt x_t = a' x0 + b' x1 (flow matching).
    Any schedule.

    :meth:`to_score` divides by g^2, which vanishes as t -> 0 on VE and flow
    matching paths; it only runs at eta2 > 0, and the grids stop at t_min.
    """

    velocity_needs_chart = False  # the head *is* the velocity

    def target(self, schedule, x0, x1, t):
        a_dot = expand_t(schedule.a_dot(t), x0)
        b_dot = expand_t(schedule.b_dot(t), x1)
        return a_dot * x0 + b_dot * x1

    def to_score(self, schedule, output, x_t, t):
        # Invert v = f x - 1/2 g^2 s.
        sde = SDE(schedule)
        f = expand_t(sde.f(t), x_t)
        g2 = expand_t(sde.g2(t), x_t)
        return 2.0 * (f * x_t - output) / g2

    def to_velocity(self, schedule, output, x_t, t):
        return output


class PseudoForceParametrization(Parametrization):
    """
    The head predicts the pseudo force F = 2 (x0 - x_t) (GPFF convention).
    Any schedule.

    Recovering x0 = x_t + F/2 needs neither division nor sigma, which is what
    :class:`~schnetpack.dynamics.optimize.DirectDenoising` relies on. On a VE
    path the target is -2 b x1, so |F| carries the noise level and the head
    needs no time input; the target's scale then runs with b, so pass
    ``weight=lambda t: (1 / schedule.b(t)**2).clamp(max=1.0)`` to the loss.
    """

    def target(self, schedule, x0, x1, t):
        a = expand_t(schedule.a(t), x0)
        b = expand_t(schedule.b(t), x1)
        return 2.0 * ((1.0 - a) * x0 - b * x1)

    def to_x0(self, schedule, output, x_t, t):
        # Direct, and exact at b = 0 — the definition of F rearranged.
        return x_t + 0.5 * output

    def to_pseudo_force(self, schedule, output, x_t, t):
        return output

    def to_score(self, schedule, output, x_t, t):
        # Tweedie on the recovered x0; for VE this reduces to F / (2 sigma^2).
        a = expand_t(schedule.a(t), x_t)
        sigma = expand_t(schedule.sigma(t), x_t)
        return (a * self.to_x0(schedule, output, x_t, t) - x_t) / sigma**2


class ParametrizedModel(nn.Module):
    """
    A model with a raw head, read as the fields a calculator expects.

    Runs ``model`` and converts its raw head ``outputs[output_key]`` with the
    parametrization it was trained under into every field whose key is set,
    written next to the model's own outputs. Fields whose key is None are not
    computed. This is where the conversions live, so the
    :class:`~schnetpack.dynamics.calculator.GenerativeCalculator` subclasses
    can read their field straight from the outputs.
    """

    def __init__(
        self,
        model: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        parametrization: Parametrization,
        schedule: Schedule,
        output_key: str = "prediction",
        key: str = properties.R,
        time_key: str = properties.t,
        score_key: str | None = None,
        noise_key: str | None = None,
        velocity_key: str | None = None,
        x0_key: str | None = None,
        pseudo_force_key: str | None = None,
    ):
        """
        Args:
            model: batch -> outputs, with the raw head under ``output_key``
            parametrization: contract the model was trained under
            schedule: schedule the model was trained on
            output_key: model output holding the raw head
            key: batch key holding x_t
            time_key: batch key holding the path time
            score_key: output key for the score, or None
            noise_key: output key for the expected unit noise, or None
            velocity_key: output key for the velocity, or None
            x0_key: output key for the denoised sample, or None
            pseudo_force_key: output key for the pseudo force, or None

        Raises:
            ValueError: if no field key is set.
        """
        super().__init__()
        parametrization.validate(schedule)
        conversions = {
            score_key: parametrization.to_score,
            noise_key: parametrization.to_noise,
            velocity_key: parametrization.to_velocity,
            x0_key: parametrization.to_x0,
            pseudo_force_key: parametrization.to_pseudo_force,
        }
        conversions.pop(None, None)
        if not conversions:
            raise ValueError("ParametrizedModel needs at least one field key")
        self.model = model
        self.parametrization = parametrization
        self.schedule = schedule
        self.output_key = output_key
        self.key = key
        self.time_key = time_key
        self.conversions = conversions

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = dict(self.model(inputs))
        raw = outputs[self.output_key]
        x_t, t = inputs[self.key], inputs[self.time_key]
        for field_key, convert in self.conversions.items():
            outputs[field_key] = convert(self.schedule, raw, x_t, t)
        return outputs

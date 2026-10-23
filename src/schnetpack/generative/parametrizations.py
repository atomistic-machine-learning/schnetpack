"""
Parametrizations: what the network predicts, and the conversions that follow.

A parametrization is stateless field math; every method takes the
:class:`~schnetpack.generative.schedules.Schedule` it applies to. It owns the
training target (:meth:`Parametrization.target`) and the conversions of a raw
output into the canonical fields (``to_score``, ``to_velocity``, ``to_x0``).
Consumers that pair it with a schedule call :meth:`Parametrization.validate`
at construction. The model stays a bare callable that returns the raw head;
the sampler's updates convert it with the parametrization of its
:class:`~schnetpack.generative.variables.Variable`.
Theory and the catalog: ``docs_new/parametrizations.md``.
"""

import abc

import torch
import torch.nn.functional as F

from schnetpack.generative.noising import expand_t
from schnetpack.generative.priors import MASK
from schnetpack.generative.schedules import GaussianSchedule, Schedule
from schnetpack.generative.sde import SDE

__all__ = [
    "Parametrization",
    "ScoreParametrization",
    "EpsParametrization",
    "X0Parametrization",
    "VelocityParametrization",
    "PseudoForceParametrization",
    "LogitsParametrization",
    "MaskedTypeWeight",
    "SNRWeight",
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

    False only for a head that predicts the velocity itself; an
    :class:`~schnetpack.dynamics.sample.sampler.ODEUpdate` then steps it on
    any interpolant.
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


class LogitsParametrization(Parametrization):
    """
    The head predicts logits over the classes of a masked type, class index =
    Z and :data:`~schnetpack.generative.priors.MASK` = 0.

    Read as the denoiser p(z0 | z_t): a softmax with MASK's logit at -inf for
    a masked atom, since no clean type is masked, and one-hot at z_t for a
    revealed one. The loss is the cross-entropy of the same softmax. A masked
    variable has no score, velocity or noise; it is stepped by unmasking.
    Theory: ``docs_new/generation/new_variables.md`` §4.2–4.3.
    """

    def __init__(self, n_classes: int = 101):
        """
        Args:
            n_classes: number of classes, MASK included; Z < n_classes
        """
        self.n_classes = n_classes

    def target(self, schedule, x0, x1, t):
        # the clean types; the loss weight decides which atoms count
        return x0

    def log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """log p(z0 | z_t) of a masked atom: log-softmax with MASK at -inf."""
        logits = logits.clone()
        logits[..., MASK] = float("-inf")
        return logits.log_softmax(-1)

    def loss(
        self, logits: torch.Tensor, label: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy weighted per atom, summed over the atoms with a positive
        weight and divided by their number.

        Args:
            logits: the head's output, shape (n_atoms, n_classes)
            label: clean types, shape (n_atoms,)
            weight: loss weight per atom, 0 for atoms that do not count
        """
        counted = weight > 0
        ce = F.nll_loss(self.log_probs(logits), label, reduction="none")
        weighted = torch.where(counted, weight * ce, torch.zeros_like(ce))
        return weighted.sum() / counted.sum().clamp(min=1)

    def to_x0(self, schedule, output, x_t, t):
        """p(z0 | z_t), shape (n_atoms, n_classes)."""
        probs = self.log_probs(output).exp()
        known = F.one_hot(x_t, self.n_classes).to(probs)
        return torch.where((x_t == MASK)[:, None], probs, known)

    def to_score(self, schedule, output, x_t, t):
        raise TypeError(
            "a masked type has no score, velocity or noise; read it as x0 and "
            "step it with AncestralUpdate"
        )

    def to_pseudo_force(self, schedule, output, x_t, t):
        return self.to_score(schedule, output, x_t, t)


class MaskedTypeWeight:
    """
    Loss weight of a masked type: the KL of one reverse step per unit
    cross-entropy, plus ``lambda_ce``, on masked atoms; 0 on revealed ones.

    The step of a T-step sampler reveals a masked atom with probability
    -a'(t) / (T (1 - a(t))), which is 1 / (T t) for a = 1 - t: GEODE's Eq. 16.
    Use as a :class:`~schnetpack.generative.variables.Variable` weight.
    Theory: ``docs_new/generation/new_variables.md`` §4.3.
    """

    def __init__(self, n_steps: int = 1000, lambda_ce: float = 0.005):
        """
        Args:
            n_steps: the T of the reverse step
            lambda_ce: weight of the plain cross-entropy
        """
        self.n_steps = n_steps
        self.lambda_ce = lambda_ce

    def __call__(self, schedule, t, z_t):
        """
        Args:
            schedule: the masking schedule, with a(t) the keep probability
            t: times per atom, shape (n_atoms,)
            z_t: noisy types, shape (n_atoms,)

        Returns:
            Weight per atom, shaped like t.
        """
        reveal = -schedule.a_dot(t) / (self.n_steps * (1.0 - schedule.a(t)))
        # where, not a product: at a(t) = 1 no atom is masked, but reveal is inf
        return torch.where(z_t == MASK, reveal + self.lambda_ce, torch.zeros_like(t))


class SNRWeight:
    """
    Loss weight a(t)^2 / sigma(t)^2, the signal-to-noise ratio.

    An x0 head on a Gaussian interpolant trained with it has the loss of an
    eps head, ||eps_hat - eps||^2 = (a / sigma)^2 ||x0_hat - x0||^2, with
    eps_hat the noise its x0_hat implies. GEODE trains its cell this way
    (its Eq. 8 and 9). Use as a
    :class:`~schnetpack.generative.variables.Variable` weight. Theory:
    ``docs_new/generation/new_variables.md`` §5.3.
    """

    def __call__(self, schedule, t, x_t):
        """
        Args:
            schedule: a Gaussian schedule
            t: times per row, shape (n_rows,)
            x_t: noisy rows; unused

        Returns:
            Weight per row, shaped like t.
        """
        return (schedule.a(t) / schedule.sigma(t)) ** 2

"""
Differential-equation representations of a schedule and their reversal.

- :class:`SDE`: the (f, g) chart of a Gaussian schedule, the linear forward
  SDE dx = f x dt + g dw with the same marginals as the Gaussian
  interpolant, plus its closed forms (perturbation kernel, exact posterior).
- :class:`ReverseSDE`: the Anderson family reversing that chart, from the
  reverse-time SDE (eta2 = 1) to the probability-flow ODE (eta2 = 0).
- :class:`ReverseODE`: transport along a learned velocity, dx = v dt, valid
  for any endpoint law and independent of the chart.

Both reverse classes are coefficient math only: ``drift`` takes the field —
the score for :class:`ReverseSDE`, the velocity for :class:`ReverseODE` — as
an input, so evaluating the model stays with the driver. Both expose
``drift``/``diffusion`` for steppers that run backwards in time (dt < 0).
Theory: ``docs_new/generation/generation.md``.
"""

import torch

from schnetpack.generative.noising import expand_t
from schnetpack.generative.schedules import GaussianSchedule

__all__ = ["SDE", "ReverseSDE", "ReverseODE"]


class SDE:
    """
    The (f, g) chart of a Gaussian schedule: the linear forward SDE whose
    conditional marginals N(a x0, sigma^2 I) match those of a
    :class:`~schnetpack.generative.noising.GaussianInterpolant` with this
    schedule.
    """

    def __init__(self, schedule: GaussianSchedule):
        """
        Args:
            schedule: the Gaussian schedule a(t), sigma(t)

        Raises:
            TypeError: if the schedule is not a GaussianSchedule.
        """
        if not isinstance(schedule, GaussianSchedule):
            raise TypeError(
                f"No (f, g) SDE chart for {type(schedule).__name__}: it needs "
                "a GaussianSchedule, which defines sigma(t). The chart-free "
                "routes remain available: velocity sampling at eta2 = 0 and "
                "the direct x0/pseudo-force recovery (DirectDenoising)."
            )
        self.schedule = schedule

    @classmethod
    def reverse(cls, schedule: GaussianSchedule, eta2: float = 1.0) -> "ReverseSDE":
        """
        The reverse of the schedule's forward SDE, as a :class:`ReverseSDE`.

        Args:
            schedule: the Gaussian schedule a(t), sigma(t)
            eta2: stochasticity in [0, 1]; 1 = reverse SDE, 0 =
                probability-flow ODE
        """
        return ReverseSDE(cls(schedule), eta2=eta2)

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient f(t) = d/dt log a."""
        return self.schedule.log_a_dot(t)

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion g^2(t) = 2 sigma (sigma' - f sigma).
        """
        sigma = self.schedule.sigma(t)
        return 2.0 * sigma * (self.schedule.sigma_dot(t) - self.f(t) * sigma)

    def posterior(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and std of the exact Gaussian posterior p(x_s | x_t, x0), s < t.

        With r = a_t / a_s and var_ts = sigma_t^2 - r^2 sigma_s^2:
        mean = (r sigma_s^2 / sigma_t^2) x_t + (a_s var_ts / sigma_t^2) x0
        and var = sigma_s^2 var_ts / sigma_t^2. This is the step ancestral
        sampling and DDIM discretize.

        Args:
            x_t: states at time t
            x0: clean data, or the model's estimate of it
            t: current times, per-sample or scalar
            s: target times, s < t elementwise

        Returns:
            (mean, std): mean shaped like x_t, std shaped like t.
        """
        a_t, sig_t = self.schedule.a(t), self.schedule.sigma(t)
        a_s, sig_s = self.schedule.a(s), self.schedule.sigma(s)
        r = a_t / a_s
        var_ts = sig_t**2 - r**2 * sig_s**2
        mean = (
            expand_t(r * sig_s**2 / sig_t**2, x_t) * x_t
            + expand_t(a_s * var_ts / sig_t**2, x0) * x0
        )
        std = torch.sqrt(torch.clamp(sig_s**2 * var_ts / sig_t**2, min=0.0))
        return mean, std

    def x0_from_score(
        self, x_t: torch.Tensor, score: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Tweedie's formula: x0 = (x_t + sigma^2 score) / a.

        Singular where a -> 0 (flow matching at t = 1, VP at large t).
        """
        a, sigma = self.schedule.a(t), self.schedule.sigma(t)
        return (x_t + expand_t(sigma**2, x_t) * score) / expand_t(a, x_t)


class ReverseSDE:
    """
    Time reversal of a forward SDE as a one-parameter family,

        dx = [f x - 1/2 (1 + eta2) g^2 score(x, t)] dt + sqrt(eta2) g dw,

    eta2 in [0, 1]: 1 is the Anderson (1982) reverse-time SDE, 0 the
    probability-flow ODE, and every member shares the forward marginals.
    eta2 is the square of the eta knob of DDIM. Takes the :class:`SDE` chart
    only; the score is an input to :meth:`drift`, and composing model,
    parametrization and conditioning into it is the caller's job
    (:class:`~schnetpack.dynamics.calculator.ScoreCalculator`).
    """

    def __init__(self, sde: SDE, eta2: float = 1.0):
        """
        Args:
            sde: the (f, g) chart of the forward process being reversed
            eta2: stochasticity in [0, 1]; 1 = reverse SDE, 0 =
                probability-flow ODE
        """
        self.sde = sde
        self.eta2 = eta2

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """Squared diffusion of the process being reversed."""
        return self.sde.g2(t)

    def drift(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """
        Anderson drift f x - 1/2 (1 + eta2) g^2 score.

        Args:
            x: current state
            t: current time, per-sample or scalar
            score: score of the marginal p_t at (x, t)
        """
        f = expand_t(self.sde.f(t), x)
        g2 = expand_t(self.g2(t), x)
        return f * x - 0.5 * (1.0 + self.eta2) * g2 * score

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion sqrt(eta2) g(t), shaped like t; zero on the ODE."""
        if self.eta2 == 0.0:
            return torch.zeros_like(t)
        return torch.sqrt(self.eta2 * self.g2(t))


class ReverseODE:
    """
    Deterministic reverse transport along a learned velocity: dx = v dt.

    Valid for any endpoint law and never touches the :class:`SDE` chart, so
    the velocity handed to :meth:`drift` must not cross it either (a velocity
    head, see
    :attr:`~schnetpack.generative.parametrizations.Parametrization.velocity_needs_chart`).
    """

    eta2 = 0.0
    """Deterministic: the ODE is the eta2 = 0 member."""

    def drift(
        self, x: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        The drift is the velocity itself.

        Args:
            x: current state
            t: current time, per-sample or scalar
            velocity: probability-flow velocity at (x, t)
        """
        return velocity

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Zero: this is the ODE."""
        return torch.zeros_like(t)

"""
The time-indexed family: drivers that solve a generative model's reverse
process along a time grid.

- :class:`SDESampler` integrates the reverse SDE of a Gaussian forward
  process on the score of a
  :class:`~schnetpack.dynamics.calculator.ScoreCalculator`.
- :class:`ODESampler` integrates dx = v dt on the velocity of a
  :class:`~schnetpack.dynamics.calculator.VelocityCalculator`.
- :class:`AncestralSampler` steps with the exact Gaussian posterior of the
  forward process on the denoised sample of an
  :class:`~schnetpack.dynamics.calculator.X0Calculator`.

All share the loop of :class:`Sampler`, and the time range is the grid's.
The SDE and ODE samplers take their step rule as an
:class:`~schnetpack.dynamics.sample.integrators.Integrator`.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import (
    ScoreCalculator,
    VelocityCalculator,
    X0Calculator,
)
from schnetpack.dynamics.sample.grids import TimeGrid, UniformGrid
from schnetpack.dynamics.sample.integrators import Euler, Integrator
from schnetpack.generative.differential_equations import SDE
from schnetpack.generative.noising import GaussianInterpolant, expand_t
from schnetpack.generative.priors import Prior

__all__ = ["Sampler", "SDESampler", "ODESampler", "AncestralSampler"]


class Sampler(Dynamics):
    """
    Shared loop of the samplers: prior -> :meth:`step` along a time grid.

    Hooks run between steps with ``batch[time_key]`` the grid time of the
    iterate they see. The batch is in the model's units. Guidance belongs to
    the calculator: the field it returns is already guided.
    """

    def __init__(
        self,
        calculator,
        grid: TimeGrid,
        prior: Prior | None = None,
        hooks: Sequence = (),
    ):
        """
        Args:
            calculator: the model's field
            grid: where to place the steps, and the time range they cover
            prior: starting distribution :meth:`sample` draws from; without
                one, only :meth:`run` on given structures is available
            hooks: run around every step, in order
        """
        super().__init__(calculator, prior=prior, hooks=hooks, key=calculator.key)
        self.grid = grid

    @property
    def time_key(self) -> str:
        """Batch key the path time is written to, the calculator's."""
        return self.calculator.time_key

    def run(self, batch: Mapping[str, Any], n_steps: int) -> dict[str, Any]:
        """
        Move the structures in ``batch`` along the grid, from its ``t_start``
        to its ``t_end``.

        Args:
            batch: structures sitting at the grid's ``t_start``
            n_steps: number of steps

        Returns:
            The final batch.
        """
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        x = batch[self.key]
        ts = self.grid(n_steps, dtype=x.dtype, device=x.device)
        n_steps = ts.shape[0] - 1
        n_rows = x.shape[0]

        batch = {**batch, self.time_key: ts[0].expand(n_rows)}
        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)
            x = self.step(
                batch, batch[self.key], batch[self.time_key], ts[i + 1] - ts[i]
            )
            batch = {**batch, self.key: x, self.time_key: ts[i + 1].expand(n_rows)}
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

    @abc.abstractmethod
    def step(
        self,
        batch: Mapping[str, Any],
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance x from t to t + dt.

        Args:
            batch: the current batch, carried along to every field evaluation
            x: current state, shape (n_rows, ...)
            t: current time, shape (n_rows,)
            dt: time increment (0-dim tensor; negative when denoising)

        Returns:
            The advanced state.
        """
        raise NotImplementedError


def _check_gaussian(sampler, forward_process) -> None:
    """Raise unless the sampler gets a Gaussian forward process."""
    if not isinstance(forward_process, GaussianInterpolant):
        raise TypeError(
            f"{type(sampler).__name__} reverses a Gaussian forward process and "
            f"needs a GaussianInterpolant; got {type(forward_process).__name__}"
        )


def _check_integrator(integrator: Integrator | None) -> Integrator:
    """The integrator, defaulting to Euler; raise unless it is one."""
    if integrator is None:
        return Euler()
    if not isinstance(integrator, Integrator):
        raise TypeError(f"{type(integrator).__name__} is not an Integrator")
    return integrator


class SDESampler(Sampler):
    """
    Reverse SDE of a Gaussian forward process, integrated on the score:

        dx = [f x - 1/2 (1 + eta2) g^2 score] dt + sqrt(eta2) g dw,

    eta2 = 1 the reverse-time SDE and eta2 = 0 the probability-flow ODE.
    Its coefficients are :attr:`reverse_sde`, built as
    ``SDE.reverse(schedule, eta2)``. The integrator reads them through
    :meth:`drift` and :meth:`diffusion`.
    """

    def __init__(
        self,
        calculator: ScoreCalculator,
        forward_process: GaussianInterpolant,
        eta2: float = 1.0,
        integrator: Integrator | None = None,
        grid: TimeGrid | None = None,
        prior: Prior | None = None,
        hooks: Sequence = (),
    ):
        """
        Args:
            calculator: the model's score
            forward_process: Gaussian interpolant the model was trained on
            eta2: stochasticity of the reverse process; 1 = reverse SDE,
                0 = probability-flow ODE
            integrator: step rule (default:
                :class:`~schnetpack.dynamics.sample.integrators.Euler`)
            grid: where to place the steps (default: uniform from the
                schedule's t_max to its t_min)
            prior: starting distribution (default: the forward process's
                prior)
            hooks: run around every step, in order
        """
        if not isinstance(calculator, ScoreCalculator):
            raise TypeError(
                f"{type(self).__name__} steps on the score and needs a "
                f"ScoreCalculator; got {type(calculator).__name__}"
            )
        _check_gaussian(self, forward_process)
        schedule = forward_process.schedule
        if grid is None:
            grid = UniformGrid(schedule.t_max, schedule.t_min)
        if prior is None:
            prior = forward_process.prior
        super().__init__(calculator, grid=grid, prior=prior, hooks=hooks)
        self.forward_process = forward_process
        self.eta2 = eta2
        self.integrator = _check_integrator(integrator)
        self.reverse_sde = SDE.reverse(schedule, eta2)

    def drift(self, batch, x, t):
        """Anderson drift f x - 1/2 (1 + eta2) g^2 score at (x, t)."""
        return self.reverse_sde.drift(x, t, self.calculator.field(batch, x, t))

    def diffusion(self, t):
        """sqrt(eta2) g(t); None on the probability-flow ODE (eta2 = 0)."""
        if self.eta2 == 0.0:
            return None
        return self.reverse_sde.diffusion(t)

    def step(self, batch, x, t, dt):
        return self.integrator.step(self, batch, x, t, dt)


class ODESampler(Sampler):
    """
    Deterministic transport along a learned velocity, dx = v dt.

    Needs no forward process: the time range is the grid's and the starting
    distribution is passed explicitly.
    """

    def __init__(
        self,
        calculator: VelocityCalculator,
        integrator: Integrator | None = None,
        grid: TimeGrid | None = None,
        prior: Prior | None = None,
        hooks: Sequence = (),
    ):
        """
        Args:
            calculator: the model's velocity
            integrator: step rule (default:
                :class:`~schnetpack.dynamics.sample.integrators.Euler`)
            grid: where to place the steps (default: uniform from 1 to 0)
            prior: starting distribution :meth:`sample` draws from; without
                one, only :meth:`run` on given structures is available
            hooks: run around every step, in order
        """
        if not isinstance(calculator, VelocityCalculator):
            raise TypeError(
                f"{type(self).__name__} steps on the velocity and needs a "
                f"VelocityCalculator; got {type(calculator).__name__}"
            )
        if grid is None:
            grid = UniformGrid()
        super().__init__(calculator, grid=grid, prior=prior, hooks=hooks)
        self.integrator = _check_integrator(integrator)

    def drift(self, batch, x, t):
        """The velocity at (x, t)."""
        return self.calculator.field(batch, x, t)

    def diffusion(self, t):
        """None: the transport is deterministic."""
        return None

    def step(self, batch, x, t, dt):
        return self.integrator.step(self, batch, x, t, dt)


class AncestralSampler(Sampler):
    """
    Exact-posterior steps on the denoised sample of a Gaussian forward process.

    Each step reads x0_hat from the calculator and draws
    x_s ~ p(x_s | x_t, x0_hat), the Gaussian posterior of the forward
    process (:meth:`posterior`). Not a discretization of the reverse SDE but
    its exact transition given x0_hat: the DDPM step on VP, the NCSN/GPFF
    ancestral sampler on VE. Intrinsically stochastic.
    """

    def __init__(
        self,
        calculator: X0Calculator,
        forward_process: GaussianInterpolant,
        grid: TimeGrid | None = None,
        prior: Prior | None = None,
        hooks: Sequence = (),
    ):
        """
        Args:
            calculator: the model's denoised sample x0
            forward_process: Gaussian interpolant the model was trained on
            grid: where to place the steps (default: uniform from the
                schedule's t_max to its t_min)
            prior: starting distribution (default: the forward process's
                prior)
            hooks: run around every step, in order
        """
        if not isinstance(calculator, X0Calculator):
            raise TypeError(
                f"{type(self).__name__} steps on the denoised sample and needs "
                f"an X0Calculator; got {type(calculator).__name__}"
            )
        _check_gaussian(self, forward_process)
        schedule = forward_process.schedule
        if grid is None:
            grid = UniformGrid(schedule.t_max, schedule.t_min)
        if prior is None:
            prior = forward_process.prior
        super().__init__(calculator, grid=grid, prior=prior, hooks=hooks)
        self.forward_process = forward_process
        self.schedule = schedule

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
        and var = sigma_s^2 var_ts / sigma_t^2.

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

    def step(self, batch, x, t, dt):
        x0_hat = self.calculator.field(batch, x, t)
        mean, std = self.posterior(x, x0_hat, t, t + dt)
        return mean + expand_t(std, x) * torch.randn_like(x)

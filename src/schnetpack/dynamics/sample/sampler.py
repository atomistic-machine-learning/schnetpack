"""
The time-indexed family: drivers that solve a generative model's reverse
process from t_max down to t_min.

:class:`Sampler` owns the loop along the time grid and the reverse process's
drift and diffusion; a subclass is one step rule on them —
:class:`EulerMaruyama`, :class:`Heun`, :class:`Ancestral`,
:class:`AncestralDDPM`. The model, the process and the parametrization come
from one :class:`~schnetpack.dynamics.calculator.GenerativeCalculator`.
Details: ``docs_new/sampling.md`` §4.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import GenerativeCalculator
from schnetpack.dynamics.sample.grids import TimeGrid, UniformGrid
from schnetpack.generative.differential_equations import ReverseODE, ReverseSDE
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process, expand_t

__all__ = ["Sampler", "EulerMaruyama", "Heun", "Ancestral", "AncestralDDPM"]


class Sampler(Dynamics):
    """
    Base class of the samplers: prior -> reverse process -> step rule along a
    time grid.

    The reverse process is the Anderson family

        dx = [f x - 1/2 (1 + eta2) g^2 score] dt + sqrt(eta2) g dw,

    eta2 = 1 the reverse-time SDE and eta2 = 0 the probability-flow ODE.
    Its coefficients are :attr:`reverse`: a
    :class:`~schnetpack.generative.differential_equations.ReverseSDE` on the
    calculator's process, or at eta2 = 0 with a velocity head the chart-free
    :class:`~schnetpack.generative.differential_equations.ReverseODE`, so
    flow-matching sampling never crosses the (f, g) chart. :meth:`drift`
    asks the calculator for the score or the velocity and hands it to
    :attr:`reverse`; a subclass implements :meth:`step` on :meth:`drift` and
    :meth:`diffusion`.

    The starting distribution defaults to the process's sampling prior. One
    step of the loop is one :meth:`step`; state constraints run between steps
    with ``batch[time_key]`` the grid time of the iterate they see.

    The batch is in the model's units (see
    :class:`~schnetpack.dynamics.calculator.GenerativeCalculator`). Guidance
    belongs to the calculator too: the fields it returns are already guided.
    """

    time_free = False
    """The iterate sits at a known noise level, ``batch[time_key]``."""

    needs_score: bool = False
    """Whether :meth:`step` reads the score itself, not only the drift."""

    def __init__(
        self,
        calculator: GenerativeCalculator,
        grid: TimeGrid | None = None,
        prior: Prior | None = None,
        eta2: float = 1.0,
        constraints: Sequence = (),
    ):
        """
        Args:
            calculator: the model with the process and parametrization it was
                trained under
            grid: where to place the steps (default: uniform)
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal.
            eta2: stochasticity of the reverse process; 1 = reverse SDE,
                0 = probability-flow ODE
            constraints: state constraints applied around every step, in
                order
        """
        if not isinstance(calculator, GenerativeCalculator):
            raise TypeError(
                f"{type(self).__name__} runs on a GenerativeCalculator, which "
                "knows the process and parametrization the model was trained "
                f"under; got {type(calculator).__name__}"
            )
        process = calculator.process
        super().__init__(
            calculator,
            prior=prior if prior is not None else process.sampling_prior(),
            constraints=constraints,
            key=calculator.key,
        )
        self.grid = grid if grid is not None else UniformGrid()
        self.eta2 = eta2
        # Validity settles here, not mid-run: if anything in this assembly
        # will cross the (f, g) chart — stochastic sampling, a non-velocity
        # head's conversion, a step rule that reads the score — acquire the
        # chart once now, so a configuration without it fails with the
        # obstruction named instead of sampling garbage.
        self.needs_chart = (
            eta2 > 0.0
            or calculator.parametrization.velocity_needs_chart
            or self.needs_score
        )
        self.sde = process.sde() if self.needs_chart else None
        self.reverse = (
            ReverseSDE(self.sde, eta2=eta2) if self.needs_chart else ReverseODE()
        )

    @property
    def process(self) -> Process:
        """The forward process, the calculator's."""
        return self.calculator.process

    @property
    def time_key(self) -> str:
        """Batch key the path time is written to, the calculator's."""
        return self.calculator.time_key

    # -- the loop --------------------------------------------------------- #

    def run(
        self,
        batch: Mapping[str, Any],
        n_steps: int,
        t_start: float | None = None,
    ) -> dict[str, Any]:
        """
        Denoise the structures in ``batch`` from ``t_start`` down to the
        process's ``t_min``.

        Args:
            batch: structures to denoise
            n_steps: number of steps
            t_start: path time the structures are assumed to sit at
                (default: the process's ``t_max``); the partial-denoising
                entry

        Returns:
            The final batch.
        """
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        x = batch[self.key]
        t_start = self.process.t_max if t_start is None else t_start
        ts = self.grid(
            t_start, self.process.t_min, n_steps, dtype=x.dtype, device=x.device
        )
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

    # -- the reverse process ---------------------------------------------- #

    def drift(
        self, batch: Mapping[str, Any], x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Drift of the reverse process at (x, t); one model evaluation.

        f x - 1/2 (1 + eta2) g^2 score through the chart when this assembly
        needs it, the chart-free velocity otherwise.
        """
        if self.needs_chart:
            return self.reverse.drift(x, t, self.calculator.score(batch, x, t))
        return self.reverse.drift(x, t, self.calculator.velocity(batch, x, t))

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion sqrt(eta2) g(t) of the reverse process, shaped like t."""
        return self.reverse.diffusion(t)


class EulerMaruyama(Sampler):
    """First-order step: x <- x + drift dt + g sqrt(|dt|) z; plain Euler at eta2 = 0."""

    def step(self, batch, x, t, dt):
        x_new = x + self.drift(batch, x, t) * dt
        g = expand_t(self.diffusion(t), x)
        return x_new + g * dt.abs().sqrt() * torch.randn_like(x)


class Heun(Sampler):
    """
    Second-order Heun step on the drift, plus an Euler–Maruyama diffusion
    increment. At eta2 = 0 this is the EDM (Karras et al. 2022) sampler.
    """

    def step(self, batch, x, t, dt):
        f1 = self.drift(batch, x, t)
        x_pred = x + f1 * dt
        f2 = self.drift(batch, x_pred, t + dt)
        x_new = x + 0.5 * (f1 + f2) * dt
        g = expand_t(self.diffusion(t), x)
        return x_new + g * dt.abs().sqrt() * torch.randn_like(x)


class Ancestral(Sampler):
    """
    Exact-posterior ancestral step: estimate x0 from the score
    (:meth:`~schnetpack.generative.differential_equations.SDE.x0_from_score`),
    then draw x_s ~ p(x_s | x_t, x0_hat)
    (:meth:`~schnetpack.generative.differential_equations.SDE.posterior`).

    One class for every process with a Gaussian kernel, which is checked at
    construction: the textbook DDPM step on VP, the NCSN/GPFF ancestral
    sampler on VE. Intrinsically stochastic, so it ignores ``eta2``.
    """

    needs_score = True

    def step(self, batch, x, t, dt):
        sde = self.sde
        x0_hat = sde.x0_from_score(x, self.calculator.score(batch, x, t), t)
        mean, std = sde.posterior(x, x0_hat, t, t + dt)
        return mean + expand_t(std, x) * torch.randn_like(x)


class AncestralDDPM(Sampler):
    """
    DDPM ancestral step in score form, with beta_k = g(t)^2 |dt|:

        x_{k-1} = (x_k + beta_k * score) / sqrt(1 - beta_k) + sqrt(beta_k) z.

    A discretization of the reverse VP process using the DDPM
    ``sigma_t^2 = beta_t`` variance; needs a VP-type process. Intrinsically
    stochastic, so it ignores ``eta2``.
    """

    needs_score = True

    def step(self, batch, x, t, dt):
        beta = expand_t(self.sde.g2(t), x) * dt.abs()
        mean = (x + beta * self.calculator.score(batch, x, t)) / torch.sqrt(1.0 - beta)
        return mean + beta.sqrt() * torch.randn_like(x)

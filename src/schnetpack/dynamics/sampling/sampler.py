"""
Composition of process, parametrization, integrator, grid and prior into a
sampler.
"""

from typing import Optional, Sequence

from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.sampling.grids import TimeGrid, UniformGrid
from schnetpack.dynamics.sampling.integrators.base import Integrator
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process
from schnetpack.generative.differential_equations import ReverseODE, ReverseSDE

__all__ = ["Sampler"]


class Sampler(Dynamics):
    """
    Thin composition wrapper: prior -> reverse process -> integrator.

    Takes the ``(process, parametrization)`` pair — the same pair the model
    was trained under; keeping the two sides consistent is the caller's job,
    so share the objects with the training code rather than rebuilding them.
    The pairing is checked at construction via
    :meth:`~schnetpack.generative.parametrizations.Parametrization.validate`.
    The starting distribution is not asked for by default — it *is* the
    process's sampling prior (the training prior itself, since b(t_max) = 1
    and the coupling preserves the marginal), so deriving it beats restating
    it. An explicit ``prior`` overrides that, and is required when the
    process cannot state its own start (a marginal-changing coupling).

    The model is reached through a
    :class:`~schnetpack.dynamics.calculator.Calculator`, with the raw head in
    ``outputs[output_key]`` and the time read from ``batch[time_key]``; see
    :class:`~schnetpack.dynamics.base.Dynamics` for the batch contract.
    The process, parametrization and integrator stay pure tensor math on the
    moved key, whose leading axis (atoms, for positions) is the sample axis.

    One step of the :class:`~schnetpack.dynamics.base.Dynamics` loop is one
    integrator step along the time grid; state-level constraints run between
    them, with ``batch[time_key]`` the grid time of the iterate they see.

    Method-specific behavior belongs in the composed parts. If you find
    yourself subclassing this, the logic probably belongs in a process,
    parametrization, integrator, grid or constraint — that is what the axes
    are for.
    """

    def __init__(
        self,
        calculator,
        process: Process,
        parametrization: Parametrization,
        integrator: Integrator,
        grid: Optional[TimeGrid] = None,
        prior: Optional[Prior] = None,
        churn: float = 1.0,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        constraints: Sequence = (),
        **kwargs,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.Calculator`, or a bare
                callable batch -> outputs
            process: forward process the model was trained on; supplies the
                schedule and the training prior
            parametrization: contract the model was trained under
            integrator: numerical solver for the reverse process
            grid: where to place the steps (default: uniform)
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal, where there is no data-free start to derive.
            churn: stochasticity of the reverse process; 1 = reverse SDE,
                0 = probability-flow ODE. Equals eta^2 of the Anderson family.
            t_min: time to stop integration at (default: ``process.t_min``);
                the score diverges as b -> 0
            t_max: time to start integration from (default: ``process.t_max``)
            constraints: state-level constraints applied around every step,
                in order
            **kwargs: batch contract (``key``, ``output_key``, ``time_key``),
                see :class:`~schnetpack.dynamics.base.Dynamics`
        """
        super().__init__(
            calculator,
            process,
            parametrization,
            prior=prior,
            constraints=constraints,
            **kwargs,
        )
        # Validity settles here, not mid-run: if anything in this assembly
        # will cross the (f, g) chart — stochastic sampling, a non-velocity
        # head's conversion, an ancestral integrator — acquire the chart
        # once now, so a configuration without it fails with the obstruction
        # named instead of sampling garbage. The same fact picks the reverse
        # class in step().
        self.needs_chart = (
            churn > 0.0
            or parametrization.velocity_needs_chart
            or integrator.requires_sde
        )
        if self.needs_chart:
            process.sde()
        self.integrator = integrator
        self.grid = grid if grid is not None else UniformGrid()
        self.churn = churn
        self.t_min = t_min if t_min is not None else process.t_min
        self.t_max = t_max if t_max is not None else process.t_max

    def denoise(
        self,
        batch,
        n_steps: int,
        t_start: Optional[float] = None,
    ):
        """
        Denoise the structures in ``batch`` from t_start down to t_min.

        This is the partial-denoising entry point: relaxation of given
        structures, scaffolded generation and structured priors that start
        below t_max all enter here.

        Args:
            batch: structures to denoise
            n_steps: number of integrator steps
            t_start: path time the structures are assumed to sit at
                (default: ``t_max``)

        Returns:
            The final batch.
        """
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        x = batch[self.key]
        t_start = self.t_max if t_start is None else t_start
        ts = self.grid(t_start, self.t_min, n_steps, dtype=x.dtype, device=x.device)
        n_steps = ts.shape[0] - 1
        n_rows = x.shape[0]

        batch = {**batch, self.time_key: ts[0].expand(n_rows)}
        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)
            x = self.integrator.step(
                self.reverse(batch),
                batch[self.key],
                batch[self.time_key],
                ts[i + 1] - ts[i],
            )
            batch = {**batch, self.key: x, self.time_key: ts[i + 1].expand(n_rows)}
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

    def reverse(self, batch):
        """
        The reverse process of the model at ``batch``: a ReverseSDE through the
        chart when this assembly needs it, the chart-free ReverseODE otherwise.
        The integrator evaluates it at its own (x, t) — Heun's predictor is
        not the batch's iterate — so each evaluation hands the calculator
        ``batch`` with those two keys replaced.
        """
        if self.needs_chart:

            def score_fn(x, t):
                inputs = {**batch, self.key: x, self.time_key: t}
                raw = self.calculator(inputs)[self.output_key]
                return self.parametrization.to_score(self.process, raw, x, t)

            return ReverseSDE(self.process.sde(), score_fn, churn=self.churn)

        def velocity_fn(x, t):
            inputs = {**batch, self.key: x, self.time_key: t}
            raw = self.calculator(inputs)[self.output_key]
            return self.parametrization.to_velocity(self.process, raw, x, t)

        return ReverseODE(velocity_fn)

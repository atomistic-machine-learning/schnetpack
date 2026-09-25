"""
Composition of process, parametrization, integrator, grid and prior into a
sampler. Details: ``docs_new/sampling.md`` §4.
"""

from collections.abc import Sequence

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.integrators.base import Integrator
from schnetpack.dynamics.sampling.grids import TimeGrid, UniformGrid
from schnetpack.generative.differential_equations import ReverseODE, ReverseSDE
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process

__all__ = ["Sampler"]


class Sampler(Dynamics):
    """
    Thin composition: prior -> reverse process -> integrator along a grid.

    Takes the same (process, parametrization) pair the model was trained
    under, validated at construction. The starting distribution defaults to
    the process's sampling prior. The model is reached through a
    :class:`~schnetpack.dynamics.calculator.Calculator`, with the raw head in
    ``outputs[output_key]`` and the time read from ``batch[time_key]``. One
    step of the loop is one integrator step; state constraints run between
    steps with ``batch[time_key]`` the grid time of the iterate they see.

    Method-specific behavior belongs in the composed parts (process,
    parametrization, integrator, grid, constraint), not in a subclass.
    """

    time_free = False
    """The iterate sits at a known noise level, ``batch[time_key]``."""

    def __init__(
        self,
        calculator,
        process: Process,
        parametrization: Parametrization,
        integrator: Integrator,
        grid: TimeGrid | None = None,
        prior: Prior | None = None,
        churn: float = 1.0,
        t_min: float | None = None,
        t_max: float | None = None,
        constraints: Sequence = (),
        key: str = properties.R,
        output_key: str = "prediction",
        time_key: str = properties.t,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.Calculator`, or a bare
                callable batch -> outputs
            process: forward process the model was trained on
            parametrization: contract the model was trained under
            integrator: numerical solver for the reverse process
            grid: where to place the steps (default: uniform)
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal.
            churn: stochasticity of the reverse process; 1 = reverse SDE,
                0 = probability-flow ODE
            t_min: time to stop integration at (default: ``process.t_min``)
            t_max: time to start integration from (default: ``process.t_max``)
            constraints: state-level constraints applied around every step,
                in order
            key: batch key this driver moves
            output_key: model output holding the raw head
            time_key: batch key the path time is written to, one value per
                row of the moved key (the key
                :class:`~schnetpack.generative.transforms.Diffuse` wrote in
                training)
        """
        parametrization.validate(process)
        super().__init__(
            calculator,
            prior=prior if prior is not None else process.sampling_prior(),
            constraints=constraints,
            key=key,
        )
        self.process = process
        self.parametrization = parametrization
        self.output_key = output_key
        self.time_key = time_key
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
        t_start: float | None = None,
    ):
        """
        Denoise the structures in ``batch`` from ``t_start`` down to ``t_min``.

        Args:
            batch: structures to denoise
            n_steps: number of integrator steps
            t_start: path time the structures are assumed to sit at
                (default: ``t_max``); the partial-denoising entry

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
        The reverse process of the model at ``batch``: a ReverseSDE through
        the chart when this assembly needs it, the chart-free ReverseODE
        otherwise. Each field evaluation hands the calculator ``batch`` with
        the moved key and the time replaced by the integrator's own (x, t).
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

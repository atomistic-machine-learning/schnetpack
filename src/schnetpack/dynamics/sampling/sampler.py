"""
Composition of process, parametrization, integrator, grid and prior into a
sampler.
"""

from collections.abc import Sequence

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.integrators.base import Integrator
from schnetpack.dynamics.observers import SamplingFrame
from schnetpack.dynamics.sampling.grids import TimeGrid, UniformGrid
from schnetpack.generative.differential_equations import ReverseODE, ReverseSDE
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process, expand_t
from schnetpack.units import convert_units

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
    ``outputs[output_key]`` and the time read from ``batch[time_key]``;
    conditioning keys are simply left in the batch. See
    :class:`~schnetpack.dynamics.base.Dynamics` for the rest of the batch
    contract.
    The process, parametrization and integrator stay pure tensor math on the
    moved key, whose leading axis (atoms, for positions) is the sample axis.

    One step of the :class:`~schnetpack.dynamics.base.Dynamics` loop is one
    integrator step along the time grid; state-level constraints run between
    them, with ``batch[time_key]`` the grid time of the iterate they see.

    Field constraints — restraints such as
    :class:`~schnetpack.dynamics.constraints.field.HarmonicRestraint` —
    guide the score: their forces F, scaled by ``guidance_weight`` w (1/kT),
    are added to it, score + w F, which tilts the sampled density by
    exp(-w E). The term enters the bound score field itself, so the drift and
    the ancestral steps that read the score directly both see it. A velocity
    head reaches it through the chart, v - 1/2 g^2 w F, so field constraints
    require the process's (f, g) chart even there. The restraint is evaluated
    at the noisy iterate x_t, not at a clean estimate: an approximation that
    is exact only as t -> t_min.

    Method-specific behavior belongs in the composed parts. If you find
    yourself subclassing this, the logic probably belongs in a process,
    parametrization, integrator, grid or constraint — that is what the axes
    are for.
    """

    time_free = False
    """The iterate sits at a known noise level: ``batch[time_key]`` is the
    grid time, so constraints such as
    :class:`~schnetpack.dynamics.constraints.state.Scaffold` re-noise to it.
    """

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
        observers: Sequence = (),
        guidance_weight: float = 1.0,
        position_unit: str | float = "Ang",
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
            key: batch key this driver moves
            output_key: model output holding the raw head, in the
                parametrization
            time_key: batch key the path time is written to, one value per
                row of the moved key — the key
                :class:`~schnetpack.generative.transforms.Diffuse` wrote in
                training
            observers: :class:`~schnetpack.dynamics.observers.Observer` s
                the run reports to, e.g. a
                :class:`~schnetpack.dynamics.observers.TrajectoryRecorder`
                for the reverse-diffusion path
            guidance_weight: weight w of the field constraints' forces in
                the score, in 1/eV (1/kT)
            position_unit: length unit of the moved key; field constraints
                are evaluated in Angstrom
        """
        parametrization.validate(process)
        if integrator.requires_structure:
            raise ValueError(
                f"{type(integrator).__name__} steps per structure on a force "
                "field and cannot integrate a reverse process; run it with "
                "schnetpack.dynamics.relax.Relaxer"
            )
        super().__init__(
            calculator,
            prior=prior if prior is not None else process.sampling_prior(),
            constraints=constraints,
            key=key,
            observers=observers,
        )
        self.process = process
        self.parametrization = parametrization
        self.output_key = output_key
        self.time_key = time_key
        self.guidance_weight = guidance_weight
        self.length = convert_units(position_unit, "Angstrom")
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
        if self.needs_chart or self.field_constraints:
            # a velocity head folds field constraints in through g^2
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
        state = self.integrator.init_state(self.reverse(batch), x)
        self.start_observers(batch)
        try:
            self.report(0, n_steps == 0, self._frame(batch, 0, ts[0], n_steps == 0))
            for i in range(n_steps):
                batch = self.before_step(batch, i, n_steps)
                x, state = self.integrator.step(
                    self.reverse(batch),
                    batch[self.key],
                    batch[self.time_key],
                    ts[i + 1] - ts[i],
                    state,
                )
                batch = {
                    **batch,
                    self.key: x,
                    self.time_key: ts[i + 1].expand(n_rows),
                }
                batch = self.after_step(batch, i + 1, n_steps)
                final = i + 1 == n_steps
                self.report(i + 1, final, self._frame(batch, i + 1, ts[i + 1], final))
        finally:
            self.end_observers()
        return batch

    def _frame(self, batch, step, t, final):
        """Builder of the frame of ``batch``, called only if someone listens."""
        return lambda: SamplingFrame(
            step=step, final=final, positions=batch[self.key], batch=batch, t=t
        )

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
                score = self.parametrization.to_score(self.process, raw, x, t)
                guidance = self.guidance(inputs, x)
                return score if guidance is None else score + guidance

            return ReverseSDE(self.process.sde(), score_fn, churn=self.churn)

        def velocity_fn(x, t):
            inputs = {**batch, self.key: x, self.time_key: t}
            raw = self.calculator(inputs)[self.output_key]
            velocity = self.parametrization.to_velocity(self.process, raw, x, t)
            guidance = self.guidance(inputs, x)
            if guidance is None:
                return velocity
            g2 = expand_t(self.process.sde().g2(t), x)
            return velocity - 0.5 * g2 * guidance

        return ReverseODE(velocity_fn)

    def guidance(self, batch, x):
        """
        The field constraints' score term w F at ``x``, in the moved key's
        units, or None without field constraints.
        """
        terms = self.field_terms(batch, x * self.length)
        if terms is None:
            return None
        return (self.guidance_weight * self.length) * terms.forces.to(x)

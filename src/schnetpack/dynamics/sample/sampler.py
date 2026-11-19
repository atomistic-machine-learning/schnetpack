"""
The time-indexed family: one sampler that solves the reverse process of
every generated variable along a time grid.

- :class:`Sampler` steps a list of variables, one update each. A step is a
  double loop: for every stage, one model call at the stage time, then each
  update that has this stage reads its field and writes its next stage
  point; after the last stage every update writes its new value, in list
  order.
- An update reads its variable's raw head from the model's outputs,
  converts it with the variable's parametrization into the field it steps
  on, and adds its :mod:`~schnetpack.dynamics.guidance` terms to that
  field.
- :class:`SDEUpdate` integrates the reverse SDE of a Gaussian interpolant on
  the score, :class:`ODEUpdate` transports along a velocity; both take an
  :class:`~schnetpack.dynamics.sample.integrators.Integrator`, which sets
  their stages.
- :class:`AncestralUpdate` takes the noising's own reverse transition given
  the denoiser x0, in one stage: the exact Gaussian posterior, or unmasking
  for masked atom types.

Theory: ``docs_new/generation/generation.md`` (one variable) and
``docs_new/generation/new_variables.md`` (several).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import FieldCalculator
from schnetpack.dynamics.guidance import Guidance, guidance_terms, weighted_sum
from schnetpack.dynamics.sample.grids import TimeGrid, UniformGrid
from schnetpack.dynamics.sample.integrators import Euler, Integrator
from schnetpack.dynamics.structures import StructureProvider
from schnetpack.generative.noising import (
    GaussianInterpolant,
    Interpolant,
    MaskNoising,
    expand_t,
)
from schnetpack.generative.priors import MASK, noise_like
from schnetpack.generative.schedules import GaussianSchedule
from schnetpack.generative.sde import SDE
from schnetpack.generative.variables import Variable

__all__ = [
    "Sampler",
    "Update",
    "DriftUpdate",
    "SDEUpdate",
    "ODEUpdate",
    "AncestralUpdate",
    "gaussian_posterior",
]


def gaussian_posterior(
    schedule: GaussianSchedule,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean and std of the exact Gaussian posterior p(x_s | x_t, x0), s < t.

    With r = a_t / a_s and var_ts = sigma_t^2 - r^2 sigma_s^2,
    mean = (r sigma_s^2 / sigma_t^2) x_t + (a_s var_ts / sigma_t^2) x0 and
    var = sigma_s^2 var_ts / sigma_t^2.

    Returns:
        (mean, std): mean shaped like x_t, std shaped like t.
    """
    a_t, sig_t = schedule.a(t), schedule.sigma(t)
    a_s, sig_s = schedule.a(s), schedule.sigma(s)
    r = a_t / a_s
    var_ts = sig_t**2 - r**2 * sig_s**2
    mean = (
        expand_t(r * sig_s**2 / sig_t**2, x_t) * x_t
        + expand_t(a_s * var_ts / sig_t**2, x0) * x0
    )
    std = torch.sqrt(torch.clamp(sig_s**2 * var_ts / sig_t**2, min=0.0))
    return mean, std


class Update:
    """
    Steps one variable on one field of it.

    An update has :attr:`n_stages` stages at :attr:`nodes` (fractions of the
    step). At each of them, :meth:`stage` reads the field and writes the
    variable's next stage point; at its last one it draws the variable's new
    value, which later model calls of the step see at t_next.
    :meth:`combine` only writes that value.

    :meth:`field` reads the variable's raw head from the model's outputs,
    converts it with the variable's parametrization, and :meth:`guide` adds
    the guidance to the result as it is: field + sum_i w_i G_i, every term
    evaluated at the stage point and time and shaped like the field.
    """

    n_stages: int = 1
    nodes: tuple[float, ...] = (0.0,)

    def __init__(self, variable: Variable, guidance: Sequence[Guidance] = ()):
        """
        Args:
            variable: the variable this update steps
            guidance: terms added to the field this update steps on
        """
        self.variable = variable
        self.guidance = guidance_terms(guidance)

    def check(self) -> None:
        """Raise unless this update can step its variable."""

    def head(self, outputs: Mapping[str, Any]) -> torch.Tensor:
        """The variable's raw head in the model's outputs."""
        key = self.variable.output_key
        if key not in outputs:
            raise KeyError(
                f"{type(self).__name__} of {self.variable.name!r} reads the "
                f"head {key!r}; the model returned {sorted(outputs)}"
            )
        return outputs[key]

    def guide(self, batch: Mapping[str, Any], field: torch.Tensor) -> torch.Tensor:
        """
        ``field`` plus the weighted sum of the guidance terms at ``batch``.

        Raises:
            ValueError: if the terms are not shaped like ``field``.
        """
        if not self.guidance:
            return field
        term = weighted_sum(self.guidance, batch)
        if term.shape != field.shape:
            raise ValueError(
                f"the guidance of {self.variable.name!r} is shaped "
                f"{tuple(term.shape)}, its field {tuple(field.shape)}"
            )
        return field + term.to(field)

    def field(
        self, batch: Mapping[str, Any], outputs: Mapping[str, Any], t: torch.Tensor
    ) -> torch.Tensor:
        """
        The field this update steps on, guidance included.

        Args:
            batch: the stage point, with the time t
            outputs: the model's outputs at ``batch``
            t: the time of ``batch`` (0-dim)
        """
        raise NotImplementedError

    def stage(
        self,
        stage: int,
        batch: Mapping[str, Any],
        outputs: Mapping[str, Any],
        t_stage: torch.Tensor,
        start: Mapping[str, Any],
        t: torch.Tensor,
        t_next: torch.Tensor,
        data: Sequence[Any],
    ) -> tuple[dict[str, Any], Any]:
        """
        Read the field of one stage and move the variable on.

        Args:
            stage: stage index, 0 <= stage < n_stages
            batch: the current stage point of every variable, with the time
                ``t_stage``
            outputs: the model's outputs at ``batch``
            t_stage: the time of this stage
            start: the batch at t, the start of the step
            t: time at the start of the step (0-dim)
            t_next: time at the end of the step (0-dim)
            data: what the earlier stages of this step returned

        Returns:
            (batch, data_stage): the batch with the variable at its next
            stage point, and what this stage keeps for later stages. At the
            last stage, data_stage is the new value.
        """
        raise NotImplementedError

    def combine(self, batch: Mapping[str, Any], data: Sequence[Any]) -> dict[str, Any]:
        """Write the new value drawn at the last stage."""
        return self.variable.write(batch, data[-1])


class DriftUpdate(Update):
    """
    An update integrated along a drift by an integrator, which sets its
    stages. Subclasses give :meth:`drift` and :meth:`diffusion`.
    """

    def __init__(
        self, variable: Variable, integrator: Integrator | None = None, **kwargs
    ):
        """
        Args:
            variable: the variable this update steps
            integrator: step rule (default:
                :class:`~schnetpack.dynamics.sample.integrators.Euler`)

        Remaining keyword arguments are those of :class:`Update`.
        """
        super().__init__(variable, **kwargs)
        integrator = Euler() if integrator is None else integrator
        if not isinstance(integrator, Integrator):
            raise TypeError(f"{type(integrator).__name__} is not an Integrator")
        if len(integrator.nodes) != integrator.n_stages:
            raise ValueError(
                f"{type(integrator).__name__} has {integrator.n_stages} stages "
                f"but {len(integrator.nodes)} nodes"
            )
        self.integrator = integrator
        self.nodes = tuple(integrator.nodes)
        self.n_stages = integrator.n_stages

    def check(self):
        if not isinstance(self.variable.noising, GaussianInterpolant):
            raise TypeError(
                f"{type(self).__name__} steps a Gaussian interpolant; "
                f"{self.variable.name!r} is noised by "
                f"{type(self.variable.noising).__name__}"
            )

    def drift(self, x, t, field) -> torch.Tensor:
        raise NotImplementedError

    def diffusion(self, t) -> torch.Tensor | None:
        raise NotImplementedError

    def stage(self, stage, batch, outputs, t_stage, start, t, t_next, data):
        x, x_start = self.variable.read(batch), self.variable.read(start)
        drifts = [*data, self.drift(x, t_stage, self.field(batch, outputs, t_stage))]
        if stage + 1 < self.n_stages:
            x_next, _ = self.integrator.point(stage + 1, x_start, t, t_next, drifts)
            return self.variable.write(batch, x_next), drifts[-1]
        # the last stage: the step itself, drawn once
        x_new = self.integrator.combine(x_start, t, t_next, drifts)
        g = self.diffusion(t)
        if g is not None:
            scale = expand_t(g * (t_next - t).abs().sqrt(), x_start)
            x_new = x_new + scale * noise_like(
                self.variable.noising.prior, x_start, start
            )
        return self.variable.write(batch, x_new), x_new


class SDEUpdate(DriftUpdate):
    """
    The reverse SDE of a Gaussian interpolant on the score field,

        dx = [f x - 1/2 (1 + eta2) g^2 score] dt + sqrt(eta2) g dw,

    eta2 = 1 the reverse-time SDE and eta2 = 0 the probability-flow ODE.
    Guidance is added to the score.
    """

    def __init__(
        self,
        variable: Variable,
        eta2: float = 1.0,
        integrator: Integrator | None = None,
        **kwargs,
    ):
        """
        Args:
            variable: the variable this update steps
            eta2: stochasticity; 1 = reverse SDE, 0 = probability-flow ODE
            integrator: step rule (default: Euler)

        Remaining keyword arguments are those of :class:`Update`.
        """
        super().__init__(variable, integrator, **kwargs)
        self.eta2 = eta2
        self.reverse_sde = SDE.reverse(variable.noising.schedule, eta2)

    def field(self, batch, outputs, t):
        """The score, guidance added."""
        v = self.variable
        score = v.parametrization.to_score(
            v.noising.schedule, self.head(outputs), v.read(batch), t
        )
        return self.guide(batch, score)

    def drift(self, x, t, score):
        return self.reverse_sde.drift(x, t, score)

    def diffusion(self, t):
        return self.reverse_sde.diffusion(t)


class ODEUpdate(DriftUpdate):
    """
    Deterministic transport along the velocity field, dx = v dt. Guidance
    is added to the velocity.
    """

    def __init__(
        self, variable: Variable, integrator: Integrator | None = None, **kwargs
    ):
        """
        Args:
            variable: the variable this update steps
            integrator: step rule (default: Euler)

        Remaining keyword arguments are those of :class:`Update`.
        """
        super().__init__(variable, integrator, **kwargs)

    def check(self):
        """
        Any interpolant for a velocity head; any other head reaches the
        velocity through the (f, g) chart of a Gaussian interpolant.
        """
        noising = self.variable.noising
        if not isinstance(noising, Interpolant):
            raise TypeError(
                f"ODEUpdate steps an interpolant; {self.variable.name!r} is "
                f"noised by {type(noising).__name__}"
            )
        chart = self.variable.parametrization.velocity_needs_chart
        if chart and not isinstance(noising, GaussianInterpolant):
            raise TypeError(
                f"ODEUpdate reaches the velocity of {self.variable.name!r} "
                "through the score, which needs a Gaussian interpolant; it is "
                f"noised by {type(noising).__name__}"
            )

    def field(self, batch, outputs, t):
        """The velocity, guidance added."""
        v = self.variable
        velocity = v.parametrization.to_velocity(
            v.noising.schedule, self.head(outputs), v.read(batch), t
        )
        return self.guide(batch, velocity)

    def drift(self, x, t, velocity):
        return velocity

    def diffusion(self, t):
        return None


class AncestralUpdate(Update):
    """
    The noising's own reverse transition given the denoiser, in one stage at
    t.

    On a Gaussian interpolant, the exact posterior p(x_s | x_t, x0_hat): the
    DDPM step on VP, the NCSN step on VE. ``noise="reverse_diffusion"`` draws
    the noise with the std sqrt(sigma_t^2 - sigma_s^2) of the reverse
    diffusion instead, as GEODE does on its VE positions.

    On a :class:`~schnetpack.generative.noising.MaskNoising`, unmasking: the
    field is p(z0 | z_t), and each masked atom is revealed with probability
    (a_s - a_t) / (1 - a_t), its type drawn from the field without the
    ``excluded`` classes. Revealed types stay. The step that reaches the
    schedule's t_min reveals every remaining atom.

    Guidance is added to the field: x0_hat, or p(z0 | z_t) of a masked
    variable.
    """

    def __init__(
        self,
        variable: Variable,
        noise: str = "posterior",
        excluded: Sequence[int] = (),
        **kwargs,
    ):
        """
        Args:
            variable: the variable this update steps
            noise: "posterior" or "reverse_diffusion"; Gaussian only
            excluded: classes never drawn, e.g. atomic numbers; masked only

        Remaining keyword arguments are those of :class:`Update`.
        """
        if noise not in ("posterior", "reverse_diffusion"):
            raise ValueError(
                f"noise must be 'posterior' or 'reverse_diffusion', got {noise!r}"
            )
        super().__init__(variable, **kwargs)
        self.noise = noise
        self.excluded = sorted({int(z) for z in excluded})

    def check(self):
        noising = self.variable.noising
        if isinstance(noising, MaskNoising):
            if self.noise != "posterior":
                raise ValueError(
                    f"{self.variable.name!r} is unmasked, which draws no noise; "
                    f"noise={self.noise!r} applies to a Gaussian interpolant"
                )
            return
        if not isinstance(noising, GaussianInterpolant):
            raise TypeError(
                f"AncestralUpdate takes the posterior of a Gaussian interpolant "
                f"or unmasks a MaskNoising; {self.variable.name!r} is noised by "
                f"{type(noising).__name__}"
            )
        if self.excluded:
            raise ValueError(
                f"excluded classes apply to a masked variable; "
                f"{self.variable.name!r} is Gaussian"
            )

    def field(self, batch, outputs, t):
        """x0_hat, guidance added; p(z0 | z_t) for a masked variable."""
        v = self.variable
        x0 = v.parametrization.to_x0(
            v.noising.schedule, self.head(outputs), v.read(batch), t
        )
        return self.guide(batch, x0)

    def stage(self, stage, batch, outputs, t_stage, start, t, t_next, data):
        x_new = self.transition(batch, self.field(batch, outputs, t), t, t_next)
        return self.variable.write(batch, x_new), x_new

    def transition(
        self,
        batch: Mapping[str, Any],
        x0: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Draw x_s given x_t, the variable's state in ``batch``, and x0_hat."""
        x = self.variable.read(batch)
        if isinstance(self.variable.noising, MaskNoising):
            return self.unmask(x, x0, t, s)
        schedule = self.variable.noising.schedule
        mean, std = gaussian_posterior(schedule, x, x0, t, s)
        if self.noise == "reverse_diffusion":
            std = torch.sqrt(
                torch.clamp(schedule.sigma(t) ** 2 - schedule.sigma(s) ** 2, min=0.0)
            )
        noise = noise_like(self.variable.noising.prior, x, batch)
        return mean + expand_t(std, x) * noise

    def unmask(
        self, z: torch.Tensor, probs: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Reveal masked types from t to s.

        Args:
            z: types at t, MASK for masked atoms, shape (n_atoms,)
            probs: p(z0 | z_t), shape (n_atoms, n_classes)
            t: time at the start of the step (0-dim)
            s: time at the end of the step (0-dim)

        Returns:
            The types at s.
        """
        schedule = self.variable.noising.schedule
        if reaches(s, schedule.t_min):
            p = 1.0  # a grid ends at t_min, where a < 1: reveal the rest
        else:
            a_t, a_s = schedule.a(t), schedule.a(s)
            p = (a_s - a_t) / (1.0 - a_t)
        reveal = (z == MASK) & (torch.rand(z.shape, device=z.device) < p)
        if not reveal.any():
            return z
        probs = probs[reveal]  # a copy: the field stays as it is
        probs[:, [MASK, *self.excluded]] = 0.0
        z = z.clone()
        z[reveal] = torch.multinomial(probs, 1).squeeze(-1).to(z.dtype)
        return z


def reaches(t: torch.Tensor | float, t_min: float) -> bool:
    """Whether a grid time t is at or below t_min, up to rounding of the grid."""
    t = float(t)
    return t <= t_min or math.isclose(t, t_min, rel_tol=1e-5)


def check_updates(
    updates: Sequence[Update], variables: Sequence[Variable]
) -> tuple[float, ...]:
    """
    The stage times of a list of updates, after checking it.

    Raises:
        ValueError: unless every variable has exactly one update, or if the
            updates disagree on the time of a stage.
        TypeError: if an update cannot step its variable.
    """
    for u in updates:
        if not isinstance(u, Update):
            raise TypeError(f"{type(u).__name__} is not an Update")
    stepped = [u.variable for u in updates]
    names = [v.name for v in stepped]
    if len(set(map(id, stepped))) != len(stepped):
        raise ValueError(f"a variable has more than one update: {names}")
    missing = [v.name for v in variables if all(v is not w for w in stepped)]
    unknown = [w.name for w in stepped if all(w is not v for v in variables)]
    if missing or unknown:
        raise ValueError(
            f"every variable needs exactly one update; without one: {missing}, "
            f"not among the variables: {unknown}"
        )
    for u in updates:
        u.check()
    nodes = []
    for stage in range(max(u.n_stages for u in updates)):
        here = {u.nodes[stage] for u in updates if stage < u.n_stages}
        if any(stage >= u.n_stages for u in updates):
            here.add(1.0)  # a finished update sits at t_next
        if len(here) != 1:
            raise ValueError(
                f"the updates disagree on the time of stage {stage}: "
                f"{sorted(here)} (as fractions of the step)"
            )
        nodes.append(here.pop())
    return tuple(nodes)


class Sampler(Dynamics):
    """
    Steps several variables along one time grid, one update each.

    A step from t to t_next: for every stage, one model call at the stage
    time on the current stage point, then each update that has this stage
    converts its variable's head into its field and writes the variable's
    next stage point. After the last stage, every update writes its new
    value in list order, starting from the batch at t, through its
    variable's view. An update that has run out of stages sits at its value
    at t_next for the later model calls.

    Hooks run between steps; the batch they see carries the grid time per
    atom and per structure, as the model saw it.
    """

    def __init__(
        self,
        calculator: FieldCalculator,
        updates: Sequence[Update],
        start: Sequence[Variable],
        grid: TimeGrid | None = None,
        structures: StructureProvider | None = None,
        hooks: Sequence = (),
    ):
        """
        Args:
            calculator: one model call at time t, with the head of every
                variable
            updates: one update per variable; the order in which the new
                values are written
            start: the variables in the order their priors are drawn by
                :meth:`sample`
            grid: where to place the steps (default: uniform from the
                smallest t_max to the largest t_min of the schedules)
            structures: source of the structures :meth:`sample` starts from
            hooks: run around every step, in order
        """
        if not isinstance(calculator, FieldCalculator):
            raise TypeError(
                f"Sampler needs a FieldCalculator; got {type(calculator).__name__}"
            )
        self.updates = list(updates)
        self.start = list(start)
        if not self.updates:
            raise ValueError("Sampler needs at least one update")
        self.nodes = check_updates(self.updates, self.start)
        self.n_stages = len(self.nodes)
        if grid is None:
            schedules = [v.noising.schedule for v in self.start]
            grid = UniformGrid(
                min(s.t_max for s in schedules), max(s.t_min for s in schedules)
            )
        for v in self.start:
            if isinstance(v.noising, MaskNoising) and not reaches(
                grid.t_end, v.noising.schedule.t_min
            ):
                raise ValueError(
                    f"the grid ends at t = {grid.t_end}, above the t_min "
                    f"{v.noising.schedule.t_min} of {v.name!r}: some of its "
                    "atoms would stay masked"
                )
        super().__init__(calculator, structures=structures, hooks=hooks)
        self.grid = grid

    def _start_from(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Every variable drawn from its prior in the order of ``start``, written
        through its view: where :meth:`sample` and :meth:`sample_batches` run
        from.
        """
        for v in self.start:
            batch = v.write(batch, v.noising.sampling_prior().draw(batch))
        return batch

    def run(self, batch: Mapping[str, Any], n_steps: int) -> dict[str, Any]:
        """
        Move the structures in ``batch`` along the grid.

        Args:
            batch: structures with every variable at the grid's t_start
            n_steps: number of steps

        Returns:
            The final batch.
        """
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        states = [v.read(batch) for v in self.start]
        dtype = next((x.dtype for x in states if x.is_floating_point()), None)
        device = states[0].device
        ts = self.grid(n_steps, dtype=dtype, device=device)
        n_steps = ts.shape[0] - 1

        batch = self.calculator.with_time(batch, ts[0])
        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)
            batch = self.step(batch, ts[i], ts[i + 1])
            batch = self.calculator.with_time(batch, ts[i + 1])
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

    def step(
        self, batch: Mapping[str, Any], t: torch.Tensor, t_next: torch.Tensor
    ) -> dict[str, Any]:
        """
        Advance every variable from t to t_next.

        Args:
            batch: every variable at t
            t: current time (0-dim)
            t_next: time at the end of the step (0-dim)

        Returns:
            The batch with every variable at t_next.
        """
        start = batch_stage = dict(batch)
        data = [[] for _ in self.updates]
        for stage, node in enumerate(self.nodes):
            t_stage = (
                t if node == 0.0 else t_next if node == 1.0 else t + node * (t_next - t)
            )
            batch_stage = self.calculator.with_time(batch_stage, t_stage)
            outputs = self.calculator(batch_stage)
            for u, d in zip(self.updates, data):
                if stage < u.n_stages:
                    batch_stage, data_stage = u.stage(
                        stage, batch_stage, outputs, t_stage, start, t, t_next, d
                    )
                    d.append(data_stage)
        batch = start
        for u, d in zip(self.updates, data):
            batch = u.combine(batch, d)
        return batch

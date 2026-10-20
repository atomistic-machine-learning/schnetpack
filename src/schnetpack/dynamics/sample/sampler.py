"""
The time-indexed family: one sampler that solves the reverse process of
every generated variable along a time grid.

- :class:`Sampler` steps a list of variables, one update each. A step is a
  double loop: for every stage, one model call at the stage time, then each
  update that has this stage reads its field and writes its next stage
  point; after the last stage every update writes its new value, in list
  order.
- :class:`SDEUpdate` integrates the reverse SDE of a Gaussian interpolant on
  the score, :class:`ODEUpdate` transports along a velocity; both take an
  :class:`~schnetpack.dynamics.sample.integrators.Integrator`, which sets
  their stages.
- :class:`AncestralUpdate` takes the noising's own reverse transition given
  the denoiser, in one stage: the exact Gaussian posterior, or unmasking for
  masked atom types.

Theory: ``docs_new/generation/generation.md`` (one variable) and
``docs_new/generation/new_variables.md`` (several).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import JointCalculator
from schnetpack.dynamics.sample.grids import TimeGrid, UniformGrid
from schnetpack.dynamics.sample.integrators import Euler, Integrator
from schnetpack.dynamics.structures import StructureProvider
from schnetpack.generative.differential_equations import SDE
from schnetpack.generative.noising import (
    GaussianInterpolant,
    Interpolant,
    MaskNoising,
    expand_t,
)
from schnetpack.generative.priors import MASK, noise_like
from schnetpack.generative.schedules import GaussianSchedule
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
    Steps one variable on one field of the calculator's outputs.

    An update has :attr:`n_stages` stages at :attr:`nodes` (fractions of the
    step). At each of them, :meth:`stage` reads the field and writes the
    variable's next stage point; at its last one it draws the variable's new
    value, which later model calls of the step see at t_next.
    :meth:`combine` only writes that value.
    """

    n_stages: int = 1
    nodes: tuple[float, ...] = (0.0,)

    def __init__(self, variable: Variable, field_key: str):
        """
        Args:
            variable: the variable this update steps
            field_key: the calculator output this update reads
        """
        self.variable = variable
        self.field_key = field_key

    def check(self) -> None:
        """Raise unless this update can step its variable."""

    def field(self, fields: Mapping[str, Any]) -> torch.Tensor:
        """This update's field from the calculator's outputs."""
        if self.field_key not in fields:
            raise KeyError(
                f"{type(self).__name__} of {self.variable.name!r} reads "
                f"{self.field_key!r}; the model returned {sorted(fields)}"
            )
        return fields[self.field_key]

    def stage(
        self,
        stage: int,
        batch: Mapping[str, Any],
        fields: Mapping[str, Any],
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
            batch: the current stage point of every variable
            fields: the model's outputs at ``batch`` and ``t_stage``
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
        """Write the new value drawn at the last stage, then decode."""
        return self.variable.write(batch, data[-1])


class DriftUpdate(Update):
    """
    An update integrated along a drift by an integrator, which sets its
    stages. Subclasses give :meth:`drift` and :meth:`diffusion`.
    """

    def __init__(
        self, variable: Variable, field_key: str, integrator: Integrator | None = None
    ):
        """
        Args:
            variable: the variable this update steps
            field_key: the calculator output this update reads
            integrator: step rule (default:
                :class:`~schnetpack.dynamics.sample.integrators.Euler`)
        """
        super().__init__(variable, field_key)
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

    def stage(self, stage, batch, fields, t_stage, start, t, t_next, data):
        key = self.variable.key
        x, x_start = batch[key], start[key]
        drifts = [*data, self.drift(x, t_stage, self.field(fields))]
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
    """

    def __init__(
        self,
        variable: Variable,
        field_key: str = "score",
        eta2: float = 1.0,
        integrator: Integrator | None = None,
    ):
        """
        Args:
            variable: the variable this update steps
            field_key: the calculator output holding its score
            eta2: stochasticity; 1 = reverse SDE, 0 = probability-flow ODE
            integrator: step rule (default: Euler)
        """
        super().__init__(variable, field_key, integrator)
        self.eta2 = eta2
        self.reverse_sde = SDE.reverse(variable.noising.schedule, eta2)

    def drift(self, x, t, score):
        return self.reverse_sde.drift(x, t, score)

    def diffusion(self, t):
        return self.reverse_sde.diffusion(t)


class ODEUpdate(DriftUpdate):
    """Deterministic transport along the velocity field, dx = v dt."""

    def __init__(
        self,
        variable: Variable,
        field_key: str = "velocity",
        integrator: Integrator | None = None,
    ):
        """
        Args:
            variable: the variable this update steps
            field_key: the calculator output holding its velocity
            integrator: step rule (default: Euler)
        """
        super().__init__(variable, field_key, integrator)

    def check(self):
        """Any interpolant: the velocity needs no Gaussian chart."""
        if not isinstance(self.variable.noising, Interpolant):
            raise TypeError(
                f"ODEUpdate steps an interpolant; {self.variable.name!r} is "
                f"noised by {type(self.variable.noising).__name__}"
            )

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
    """

    def __init__(
        self,
        variable: Variable,
        field_key: str = "x0",
        noise: str = "posterior",
        excluded: Sequence[int] = (),
    ):
        """
        Args:
            variable: the variable this update steps
            field_key: the calculator output holding its x0_hat, or
                p(z0 | z_t) for a masked variable
            noise: "posterior" or "reverse_diffusion"; Gaussian only
            excluded: classes never drawn, e.g. atomic numbers; masked only
        """
        if noise not in ("posterior", "reverse_diffusion"):
            raise ValueError(
                f"noise must be 'posterior' or 'reverse_diffusion', got {noise!r}"
            )
        super().__init__(variable, field_key)
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

    def stage(self, stage, batch, fields, t_stage, start, t, t_next, data):
        x_new = self.transition(batch, self.field(fields), t, t_next)
        return self.variable.write(batch, x_new), x_new

    def transition(
        self,
        batch: Mapping[str, Any],
        x0: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Draw x_s given x_t = ``batch[key]`` and x0_hat."""
        x = batch[self.variable.key]
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
    reads its field and writes its variable's next stage point. After the
    last stage, every update writes its new value in list order, starting
    from the batch at t, with each variable's decode. An update that has
    run out of stages sits at its value at t_next for the later model calls.

    Hooks run between steps; the batch they see carries the grid time per
    atom and per structure, as the model saw it.
    """

    def __init__(
        self,
        calculator: JointCalculator,
        updates: Sequence[Update],
        start: Sequence[Variable],
        grid: TimeGrid | None = None,
        structures: StructureProvider | None = None,
        hooks: Sequence = (),
    ):
        """
        Args:
            calculator: one model call at time t, with every field the
                updates read
            updates: one update per variable; the order in which the new
                values are written
            start: the variables in the order their priors are drawn by
                :meth:`sample`
            grid: where to place the steps (default: uniform from the
                smallest t_max to the largest t_min of the schedules)
            structures: source of the structures :meth:`sample` starts from
            hooks: run around every step, in order
        """
        if not isinstance(calculator, JointCalculator):
            raise TypeError(
                f"Sampler needs a JointCalculator; got {type(calculator).__name__}"
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
        super().__init__(
            calculator, structures=structures, hooks=hooks, key=self.start[0].key
        )
        self.grid = grid

    def sample(self, n_samples: int, n_steps: int) -> dict[str, Any]:
        """
        Draw ``n_samples`` structures, then every variable from its prior in
        the order of ``start``, and run on them.
        """
        if self.structures is None:
            raise ValueError(
                "Sampler has no structures to sample from; pass structures=... "
                "at construction, or call run on given structures"
            )
        batch = self.structures(n_samples)
        for v in self.start:
            batch = v.noising.sampling_prior()(batch)
            if v.decode is not None:
                batch = v.decode(batch)
        return self.run(batch, n_steps)

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
        dtype = next(
            (
                batch[v.key].dtype
                for v in self.start
                if batch[v.key].is_floating_point()
            ),
            None,
        )
        device = batch[self.start[0].key].device
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
            fields = self.calculator.fields(batch_stage, t_stage)
            for u, d in zip(self.updates, data):
                if stage < u.n_stages:
                    batch_stage, data_stage = u.stage(
                        stage, batch_stage, fields, t_stage, start, t, t_next, d
                    )
                    d.append(data_stage)
        batch = start
        for u, d in zip(self.updates, data):
            batch = u.combine(batch, d)
        return batch

"""
Relaxation on a force field: drive structures downhill until every one of
them is relaxed.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import Calculator
from schnetpack.dynamics.integrators.base import Integrator
from schnetpack.dynamics.integrators.lbfgs import LBFGS
from schnetpack.dynamics.observers import TrajectoryRecorder
from schnetpack.dynamics.relax.observers import LogWriter, RelaxationFrame
from schnetpack.generative.priors import Prior
from schnetpack.units import convert_units

__all__ = ["Relaxer", "RelaxationResult"]


@dataclasses.dataclass
class RelaxationResult:
    """
    What a relaxation run produced.

    Attributes:
        batch: the relaxed structures, in the batch's own units
        outputs: the model outputs for them: ``energy`` in eV and ``forces``
            in eV/Angstrom under the relaxer's keys, anything else (an
            ensemble's uncertainty) as the calculator reported it
        converged: ``(n_structures,)``, which structures met ``fmax``
        n_steps: steps taken
    """

    batch: dict[str, Any]
    outputs: dict[str, Any]
    converged: torch.Tensor
    n_steps: int


class _ForceField:
    """
    The field a relaxation integrates: drift = forces, diffusion = 0.

    Works in eV and Angstrom whatever the model's units, and exposes what a
    per-structure step rule (``requires_structure``) reads: the structure
    layout and which structures are still being relaxed.
    """

    def __init__(self, relaxer: "Relaxer", batch: Mapping[str, Any]):
        self.relaxer = relaxer
        self.n_atoms = batch[properties.n_atoms]
        self.idx_m = _idx_m(batch)
        self.active = torch.ones_like(self.n_atoms, dtype=torch.bool)
        self.free = _free_atoms(batch)
        self._x = None
        self._batch = batch

    def at(self, batch: Mapping[str, Any], x: torch.Tensor) -> None:
        """Make ``x`` — ``batch``'s positions, in Angstrom — the current iterate."""
        self._batch = batch
        self._x = x

    def forces(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict]:
        """Forces on the free atoms in eV/Angstrom, and all outputs converted."""
        outputs = self.relaxer.evaluate(batch)
        forces = outputs[self.relaxer.force_key]
        if self.free is not None:
            forces = forces * self.free.to(forces.dtype)
        return forces, outputs

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x is self._x:
            # the current iterate: the very batch the relaxer checked
            # convergence on, so the calculator's cache answers
            batch = self._batch
        else:
            batch = {**self._batch, self.relaxer.key: x / self.relaxer.length}
        return self.forces(batch)[0].to(x.dtype)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


def _idx_m(batch: Mapping[str, Any]) -> torch.Tensor:
    if properties.idx_m in batch:
        return batch[properties.idx_m]
    n_atoms = batch[properties.n_atoms]
    return torch.repeat_interleave(
        torch.arange(n_atoms.shape[0], device=n_atoms.device), n_atoms
    )


def _free_atoms(batch: Mapping[str, Any]) -> torch.Tensor | None:
    """``(n_total_atoms, 1)`` float mask of the atoms that may move, or None."""
    fixed = batch.get(properties.fixed_atoms)
    if fixed is None:
        return None
    return (~fixed.to(torch.bool)).view(-1, 1)


class Relaxer(Dynamics):
    """
    Relax structures on a force field until every one of them is relaxed.

    Each step hands the force field to a step rule — an
    :class:`~schnetpack.dynamics.integrators.Integrator` — and takes the
    step it returns: :class:`~schnetpack.dynamics.integrators.LBFGS` (the
    default) for quasi-Newton relaxation, or
    :class:`~schnetpack.dynamics.integrators.EulerMaruyama` for steepest
    descent with step ``step_size``. The loop stops once the largest force on
    any free atom of every structure is below ``fmax``, or after
    ``n_steps`` steps; structures that already meet the criterion are not
    moved further.

    Everything the loop decides on is in eV and Angstrom — ``fmax``, the step
    rule's ``maxstep`` and ``step_size`` — whatever units the model works
    in; ``energy_unit`` and ``position_unit`` name the model's. The batch
    keeps its own units.

    Atoms flagged in ``batch[properties.fixed_atoms]`` are held in place:
    their forces are zeroed before the step rule sees them, their step is
    zeroed after, and they are left out of the convergence check.

    The calculator must return forces, so it is run with autograd enabled
    for models that differentiate their energy; a bare model callable is
    wrapped accordingly. The relaxer turns on the calculator's
    ``cache_last``: the convergence check and the step rule both need the
    forces at the current positions, and pay for one model call.

    Constraints run around every step as in any
    :class:`~schnetpack.dynamics.base.Dynamics`, with one difference: a run
    that converges early stops before ``n_steps``, so a constraint's final
    ``after_step(step == n_steps)`` only fires when the step limit is
    reached.
    """

    def __init__(
        self,
        calculator,
        integrator: Integrator | None = None,
        prior: Prior | None = None,
        constraints: Sequence = (),
        observers: Sequence = (),
        key: str = properties.R,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        energy_unit: str | float = "eV",
        position_unit: str | float = "Ang",
        step_size: float = 1.0,
        logfile=None,
        log_interval: int = 1,
        trajectory: str | None = None,
        trajectory_interval: int = 0,
        store_forces: bool = False,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.Calculator`, or a bare
                callable batch -> outputs
            integrator: step rule (default:
                :class:`~schnetpack.dynamics.integrators.LBFGS`)
            prior: starting distribution :meth:`sample` draws from
            constraints: state-level constraints applied around every step
            observers: :class:`~schnetpack.dynamics.observers.Observer` s the
                run reports to
            key: batch key this driver moves
            energy_key: model output holding the energy per structure
            force_key: model output holding the forces
            energy_unit: energy unit the model works in
            position_unit: length unit the model and the batch work in
            step_size: ``dt`` handed to the step rule, in Angstrom^2/eV —
                the steepest-descent step of an Euler step rule; LBFGS
                ignores it
            logfile: text progress log: a path, ``"-"`` for stdout, or None.
                Shorthand for adding a
                :class:`~schnetpack.dynamics.relax.observers.LogWriter`.
            log_interval: how often to write a log line
            trajectory: path of the HDF5 trajectory to write, or None.
                Shorthand for adding a
                :class:`~schnetpack.dynamics.observers.TrajectoryRecorder`.
            trajectory_interval: how often to write a trajectory frame
            store_forces: store the forces of every trajectory frame
        """
        if not isinstance(calculator, Calculator):
            calculator = Calculator(calculator, enable_grad=True)
        calculator.cache_last = True
        observers = list(observers)
        if logfile is not None:
            observers.append(LogWriter(logfile, interval=log_interval))
        if trajectory is not None:
            observers.append(
                TrajectoryRecorder(
                    trajectory, interval=trajectory_interval, store_forces=store_forces
                )
            )
        super().__init__(
            calculator,
            prior=prior,
            constraints=constraints,
            key=key,
            observers=observers,
        )
        self.integrator = integrator if integrator is not None else LBFGS()
        self.energy_key = energy_key
        self.force_key = force_key
        self.energy = convert_units(energy_unit, "eV")
        self.length = convert_units(position_unit, "Angstrom")
        self.step_size = step_size
        self.fmax = None

    def evaluate(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """The model outputs at ``batch``, energy and forces in eV and Angstrom."""
        outputs = self.calculator(batch)
        if self.energy_key not in outputs or self.force_key not in outputs:
            raise KeyError(
                f"the model must return {self.energy_key!r} and "
                f"{self.force_key!r} to be relaxed on; got {sorted(outputs)}"
            )
        outputs = dict(outputs)
        outputs[self.energy_key] = outputs[self.energy_key].detach() * self.energy
        outputs[self.force_key] = outputs[self.force_key].detach() * (
            self.energy / self.length
        )
        return outputs

    def _to_angstrom(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.length == 1.0 else x * self.length

    def relax(
        self, batch: Mapping[str, Any], n_steps: int, fmax: float = 0.05
    ) -> RelaxationResult:
        """
        Relax the structures in ``batch``.

        Args:
            batch: structures to relax
            n_steps: step limit
            fmax: force criterion, in eV/Angstrom

        Returns:
            The relaxed batch, the model outputs for it and which structures
            converged.
        """
        self.fmax = fmax
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        field = _ForceField(self, batch)
        n_structures = field.n_atoms.shape[0]
        x = self._to_angstrom(batch[self.key])
        state = self.integrator.init_state(field, x)
        t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        dt = torch.tensor(self.step_size, dtype=x.dtype, device=x.device)

        self.start_observers(batch)
        try:
            for i in range(n_steps + 1):
                forces, outputs = field.forces(batch)
                squared = forces.pow(2).sum(-1)
                max_sq = torch.zeros(
                    n_structures, dtype=squared.dtype, device=squared.device
                ).scatter_reduce(0, field.idx_m, squared, "amax", include_self=True)
                converged = max_sq < fmax**2
                final = bool(converged.all()) or i == n_steps
                self.report(
                    i, final, self._frame(batch, outputs, forces, max_sq, i, final)
                )
                if final:
                    break

                batch = self.before_step(batch, i, n_steps)
                x = self._to_angstrom(batch[self.key])
                field.at(batch, x)
                field.active = ~converged.to(x.device)
                x_new, state = self.integrator.step(field, x, t, dt, state)
                if field.free is not None:
                    x_new = torch.where(field.free.to(x.device), x_new, x)
                x_new = x_new if self.length == 1.0 else x_new / self.length
                batch = {**batch, self.key: x_new}
                batch = self.after_step(batch, i + 1, n_steps)
        finally:
            self.end_observers()
        return RelaxationResult(
            batch=batch, outputs=outputs, converged=converged, n_steps=i
        )

    def denoise(
        self, batch: Mapping[str, Any], n_steps: int, fmax: float = 0.05
    ) -> dict[str, Any]:
        """
        Relax the structures in ``batch``; see :meth:`relax` for the outputs.

        Args:
            batch: structures to relax
            n_steps: step limit
            fmax: force criterion, in eV/Angstrom

        Returns:
            The relaxed batch.
        """
        return self.relax(batch, n_steps, fmax=fmax).batch

    def _frame(self, batch, outputs, forces, max_sq, step, final):
        """Builder of the frame of ``batch``, called only if someone listens."""
        return lambda: RelaxationFrame(
            step=step,
            final=final,
            positions=batch[self.key],
            batch=batch,
            energy=outputs[self.energy_key],
            forces=forces,
            max_force_per_config=max_sq.sqrt(),
            fmax=self.fmax,
        )

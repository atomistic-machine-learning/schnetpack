"""
Relaxation on a force field: drive structures downhill until each of
them is relaxed.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import Calculator, ForceFieldCalculator
from schnetpack.dynamics.integrators.base import Integrator
from schnetpack.dynamics.integrators.lbfgs import LBFGS
from schnetpack.generative.priors import Prior

__all__ = ["Relaxer", "ForceField"]


class ForceField:
    """
    The field a relaxation integrates: drift = forces, diffusion = 0.

    Like :class:`~schnetpack.generative.differential_equations.ReverseODE` it
    takes a *bound field*: ``forces_fn(x) -> forces``, forces in eV/Angstrom
    on positions in Angstrom, with the model, the field constraints and the
    fixed atoms already composed inside. It also carries the structure
    layout that a per-structure step rule (``requires_structure``) reads.
    A :class:`Relaxer` builds a new one for every step
    (:meth:`Relaxer.force_field`).
    """

    def __init__(self, forces_fn: Callable, n_atoms: torch.Tensor, idx_m: torch.Tensor):
        """
        Args:
            forces_fn: bound force field, callable x -> forces
            n_atoms: ``(n_structures,)`` atoms per structure
            idx_m: ``(n_total_atoms,)`` structure of every atom
        """
        self.forces_fn = forces_fn
        self.n_atoms = n_atoms
        self.idx_m = idx_m

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The forces at ``x``. Costs one model call."""
        return self.forces_fn(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


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
    moved further, whatever the step rule. The force field the step rule
    follows is a :class:`ForceField`, built anew for every step.
    :meth:`denoise` returns the relaxed batch.

    The batch is in Angstrom, and everything the loop decides on is in eV
    and Angstrom — ``fmax``, the step rule's ``maxstep`` and ``step_size`` —
    whatever units the model works in: those are the
    :class:`~schnetpack.dynamics.calculator.ForceFieldCalculator`'s to name
    and convert. A batch in another length unit (a dataloader's, a sampler's
    output) has to be converted to Angstrom first.

    Atoms flagged in ``batch[properties.fixed_atoms]`` are held in place:
    their forces are zeroed before the step rule sees them, their step is
    zeroed after, and they are left out of the convergence check.

    The calculator must be a
    :class:`~schnetpack.dynamics.calculator.ForceFieldCalculator`, which
    returns forces under its ``force_key``; a bare model callable is wrapped
    in one, in eV and Angstrom. A plain
    :class:`~schnetpack.dynamics.calculator.Calculator` is refused: it says
    nothing about the model's units. The relaxer turns on the calculator's
    ``cache_last``: the convergence check and the step rule both need the
    forces at the current positions, and pay for one model call.

    Constraints run around every step as in any
    :class:`~schnetpack.dynamics.base.Dynamics`, with one difference: a run
    that converges early stops before ``n_steps``, so a constraint's final
    ``after_step(step == n_steps)`` only fires when the step limit is
    reached. Field constraints — restraints such as
    :class:`~schnetpack.dynamics.constraints.field.HarmonicRestraint` — are
    part of the energy surface: their forces drive the step and count
    towards ``fmax``.
    """

    def __init__(
        self,
        calculator,
        integrator: Integrator | None = None,
        prior: Prior | None = None,
        constraints: Sequence = (),
        key: str = properties.R,
        step_size: float = 1.0,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.ForceFieldCalculator`,
                or a bare callable batch -> outputs in eV and Angstrom
            integrator: step rule (default:
                :class:`~schnetpack.dynamics.integrators.LBFGS`)
            prior: starting distribution :meth:`sample` draws from
            constraints: state-level constraints applied around every step,
                and field-level constraints added to the energy surface
            key: batch key this driver moves
            step_size: ``dt`` handed to the step rule, in Angstrom^2/eV —
                the steepest-descent step of an Euler step rule; LBFGS
                ignores it
        """
        if not isinstance(calculator, Calculator):
            calculator = ForceFieldCalculator(calculator)
        elif not isinstance(calculator, ForceFieldCalculator):
            raise TypeError(
                "a Relaxer runs on a ForceFieldCalculator, which names the "
                "model's units and converts them to eV and Angstrom; got a "
                f"{type(calculator).__name__}"
            )
        calculator.cache_last = True
        super().__init__(calculator, prior=prior, constraints=constraints, key=key)
        self.integrator = integrator if integrator is not None else LBFGS()
        self.force_key = calculator.force_key
        self.step_size = step_size

    def _forces(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """
        Forces on the free atoms in eV/Angstrom.

        The field constraints' forces are part of the surface: added to the
        model's before the fixed atoms' forces are zeroed.
        """
        # detached outputs from the cache: adding the terms below neither
        # reaches the cache nor keeps the model's graph alive
        forces = self.calculator(batch)[self.force_key]
        terms = self.field_terms(batch, batch[self.key])
        if terms is not None:
            forces = forces + terms.forces.to(forces)
        fixed = batch.get(properties.fixed_atoms)
        if fixed is not None:
            forces = forces.masked_fill(fixed.unsqueeze(-1), 0.0)
        return forces

    def force_field(self, batch: Mapping[str, Any], x: torch.Tensor) -> ForceField:
        """
        The force field at ``batch``, whose iterate is ``x``.

        The step rule evaluates it at its own points — L-BFGS's are not the
        batch's iterate — so each evaluation hands the calculator ``batch``
        with the moved key replaced. At ``x`` itself it hands over ``batch``
        unchanged, the very batch the convergence check ran on, so the
        calculator's cache answers.
        """

        def forces_fn(x_eval):
            if x_eval is x:
                inputs = batch
            else:
                inputs = {**batch, self.key: x_eval}
            return self._forces(inputs).to(x_eval.dtype)

        return ForceField(forces_fn, batch[properties.n_atoms], batch[properties.idx_m])

    def denoise(
        self, batch: Mapping[str, Any], n_steps: int, fmax: float = 0.05
    ) -> dict[str, Any]:
        """
        Relax the structures in ``batch``.

        Args:
            batch: structures to relax, positions in Angstrom
            n_steps: step limit; 0 only evaluates the start
            fmax: force criterion, in eV/Angstrom

        Returns:
            The relaxed batch.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be non-negative, got {n_steps}")
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        idx_m = batch[properties.idx_m]
        fixed = batch.get(properties.fixed_atoms)
        n_structures = batch[properties.n_atoms].shape[0]
        x = batch[self.key]
        state = self.integrator.init_state(self.force_field(batch, x), x)
        t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        dt = torch.tensor(self.step_size, dtype=x.dtype, device=x.device)

        for i in range(n_steps + 1):
            squared = self._forces(batch).pow(2).sum(-1)
            max_sq = torch.zeros(
                n_structures, dtype=squared.dtype, device=squared.device
            ).scatter_reduce(0, idx_m, squared, "amax", include_self=True)
            converged = max_sq < fmax**2
            if bool(converged.all()) or i == n_steps:
                break

            batch = self.before_step(batch, i, n_steps)
            x = batch[self.key]
            x_new, state = self.integrator.step(
                self.force_field(batch, x), x, t, dt, state
            )
            # converged structures and fixed atoms stay where they are,
            # whatever the step rule
            moves = (~converged)[idx_m]
            if fixed is not None:
                moves = moves & ~fixed
            x_new = torch.where(moves.unsqueeze(-1), x_new, x)
            batch = {**batch, self.key: x_new}
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

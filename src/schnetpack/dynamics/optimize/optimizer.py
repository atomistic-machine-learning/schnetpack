"""
The time-free family: drivers that step structures along a force until they
are relaxed, or for a fixed number of steps.

:class:`Optimizer` owns the loop: the stop test, the holding of converged
structures and fixed atoms. A subclass is one step rule on the force —
:class:`GradientDescent`,
:class:`~schnetpack.dynamics.optimize.LBFGS`,
:class:`~schnetpack.dynamics.optimize.DirectDenoising`. The force comes from one
:class:`~schnetpack.dynamics.calculator.ForceCalculator`.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import Calculator, ForceCalculator
from schnetpack.dynamics.optimize.cell import generalized_forces, pack, unpack
from schnetpack.dynamics.structures import StructureProvider
from schnetpack.generative.priors import Prior

__all__ = ["Optimizer", "GradientDescent"]


class Optimizer(Dynamics):
    """
    Base class of the time-free drivers: x <- step(x, F(x)).

    Each step evaluates the force at the current structures and hands it to
    :meth:`step`, the subclass's rule. With an
    ``fmax`` the loop stops once the largest force on any free atom of every
    structure is below it, and structures that already meet the criterion
    are not moved further, whatever the step rule; without one it runs all
    ``n_steps``. :meth:`run` returns the final batch.

    The batch is in Angstrom. The force is what the
    :class:`~schnetpack.dynamics.calculator.ForceCalculator` returns: a
    physical force in eV/Angstrom, or a pseudo-force in Angstrom. ``fmax``,
    and every step-rule parameter that meets the force, is in that unit. A
    bare model callable is wrapped in a physical calculator, in eV and
    Angstrom; any other
    :class:`~schnetpack.dynamics.calculator.Calculator` is refused, since it
    says nothing about the model's units. The driver turns on the
    calculator's ``cache_last``: the stop test and the step both need the
    force at the current positions, and pay for one model call. The force
    includes the calculator's guidance — restraints such as
    :class:`~schnetpack.dynamics.guidance.HarmonicRestraint` — so a restraint
    drives the step and counts towards ``fmax``.

    Atoms flagged in ``batch[properties.fixed_atoms]`` are held in place:
    their forces are zeroed before the step rule sees them, their step is
    zeroed after, and they are left out of the stop
    test.

    With ``relax_cell`` the cell is relaxed together with the positions, from
    the calculator's stress, as ase's ``FrechetCellFilter`` does for one
    structure: each structure gets three more rows, n_atoms times the
    logarithm of its cell's deformation, and the step rule and the stop test
    treat them as three more atoms (see
    :mod:`~schnetpack.dynamics.optimize.cell`). ``fmax`` then bounds the
    force on every row, the cell's in eV, and fixed atoms keep their
    fractional coordinates. The whole cell is free; the guidance has no
    stress, so it acts on the atoms only. The loop holds the deformation
    itself: a hook may move atoms, but a cell it changes is overwritten.

    Hooks run around every step as in any
    :class:`~schnetpack.dynamics.base.Dynamics`. A run that converges early
    stops before ``n_steps``, so a hook's final
    ``after_step(step == n_steps)`` only fires when the step limit is reached.
    """

    def __init__(
        self,
        calculator,
        structures: StructureProvider | None = None,
        prior: Prior | None = None,
        hooks: Sequence = (),
        key: str = properties.R,
        fmax: float | None = None,
        relax_cell: bool = False,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.ForceCalculator`,
                or a bare callable batch -> outputs in eV and Angstrom
            structures: source of the structures :meth:`sample` starts
                from
            prior: redraws the provided structures before :meth:`sample`
                runs them; without one they are run as provided
            hooks: run around every step, in order
            key: batch key this driver moves
            fmax: stop criterion on the largest force, in the force's unit;
                None runs all ``n_steps``
            relax_cell: relax the cell together with the positions; the
                calculator needs a ``stress_key``
        """
        if not isinstance(calculator, Calculator):
            calculator = ForceCalculator(calculator)
        elif not isinstance(calculator, ForceCalculator):
            raise TypeError(
                f"{type(self).__name__} runs on a ForceCalculator, which names "
                "the model's units and converts them to eV and Angstrom; got a "
                f"{type(calculator).__name__}"
            )
        if relax_cell and calculator.stress_key is None:
            raise ValueError(
                "relax_cell steps the cell along the stress: give the "
                "ForceCalculator a stress_key"
            )
        if relax_cell and key != properties.R:
            raise ValueError(f"relax_cell moves the positions, not {key!r}")
        calculator.cache_last = True
        super().__init__(
            calculator, structures=structures, prior=prior, hooks=hooks, key=key
        )
        self.fmax = fmax
        self.relax_cell = relax_cell

    def forces(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """
        The calculator's force at ``batch``, guidance included, zero on fixed
        atoms.
        """
        forces = self.calculator.forces(batch)
        fixed = batch.get(properties.fixed_atoms)
        if fixed is not None:
            forces = forces.masked_fill(fixed.unsqueeze(-1), 0.0)
        return forces

    def _rows(
        self, batch: Mapping[str, Any], cell_rows: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The rows the step rule moves and the force on each: the positions and
        :meth:`forces`, or with ``cell_rows`` the positions and cells as
        :func:`~schnetpack.dynamics.optimize.cell.pack` lays them out.
        """
        x, forces = batch[self.key], self.forces(batch)
        if cell_rows is None:
            return x, forces.to(x.dtype)
        idx_m, n_atoms = batch[properties.idx_m], batch[properties.n_atoms]
        return (
            pack(x, cell_rows, idx_m, n_atoms),
            generalized_forces(
                forces,
                self.calculator.stress(batch),
                batch[properties.cell].view(-1, 3, 3),
                cell_rows,
                idx_m,
                n_atoms,
            ),
        )

    def _converged(
        self,
        batch: Mapping[str, Any],
        cell_rows: torch.Tensor | None,
        idx_m: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Which structures in ``batch`` are relaxed: the largest force on any of
        their free rows is below ``fmax``.

        Returns:
            ``(n_structures,)`` boolean mask, or None without an ``fmax``.
        """
        if self.fmax is None:
            return None
        squared = self._rows(batch, cell_rows)[1].pow(2).sum(-1)
        n_structures = batch[properties.n_atoms].shape[0]
        max_sq = torch.zeros(
            n_structures, dtype=squared.dtype, device=squared.device
        ).scatter_reduce(0, idx_m, squared, "amax", include_self=True)
        return max_sq < self.fmax**2

    @staticmethod
    def _moves(
        batch: Mapping[str, Any],
        converged: torch.Tensor | None,
        idx_m: torch.Tensor | None,
        n_rows: int,
    ) -> torch.Tensor | None:
        """
        Which of the ``n_rows`` rows may move: not a fixed atom, and in a
        structure not yet converged. None when every row may.
        """
        moves = None
        if converged is not None:
            moves = (~converged)[idx_m]
        fixed = batch.get(properties.fixed_atoms)
        if fixed is not None:
            # the cell rows after the atoms are never fixed
            free = torch.cat([~fixed, fixed.new_ones(n_rows - len(fixed))])
            moves = free if moves is None else moves & free
        return moves

    def run(self, batch: Mapping[str, Any], n_steps: int) -> dict[str, Any]:
        """
        Step the structures in ``batch`` until relaxed, or ``n_steps`` times.

        Args:
            batch: structures to step, positions in Angstrom
            n_steps: step limit; 0 returns the start

        Returns:
            The final batch.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be non-negative, got {n_steps}")
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        # a batch without a layout runs as long as nothing asks for one
        idx_m = batch.get(properties.idx_m)
        cell0 = cell_rows = None
        if self.relax_cell:
            n_atoms = batch[properties.n_atoms]
            # float64, like the L-BFGS history: the rows are taken from the
            # batch and written back to it on every step
            cell0 = batch[properties.cell].view(-1, 3, 3).double()
            cell_rows = cell0.new_zeros(3 * len(n_atoms), 3)
            structures = torch.arange(len(n_atoms), device=idx_m.device)
            idx_m = torch.cat([idx_m, structures.repeat_interleave(3)])
        state = self.init_state(batch, idx_m)

        for i in range(n_steps):
            converged = self._converged(batch, cell_rows, idx_m)
            if converged is not None and bool(converged.all()):
                break

            batch = self.before_step(batch, i, n_steps)
            x, forces = self._rows(batch, cell_rows)
            x_new, state = self.step(batch, x, forces, state)
            moves = self._moves(batch, converged, idx_m, len(x))
            if moves is not None:
                x_new = torch.where(moves.unsqueeze(-1), x_new, x)
            if cell_rows is None:
                batch = {**batch, self.key: x_new}
            else:
                positions, cell = unpack(x_new, cell0, batch[properties.idx_m], n_atoms)
                cell_rows = x_new[len(positions) :]
                batch = {
                    **batch,
                    properties.R: positions.to(batch[properties.R].dtype),
                    properties.cell: cell.to(batch[properties.cell].dtype).view_as(
                        batch[properties.cell]
                    ),
                }
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

    def init_state(self, batch: Mapping[str, Any], idx_m: torch.Tensor | None) -> Any:
        """
        The history the step rule carries across the steps of one run.

        Called once at loop entry; the default is no history.

        Args:
            batch: the starting batch, already prepared by the calculator
            idx_m: structure of every row the rule steps: the batch's own
                ``idx_m`` (None if it has none), with ``relax_cell`` followed
                by three cell rows per structure
        """
        return None

    @abc.abstractmethod
    def step(
        self,
        batch: Mapping[str, Any],
        x: torch.Tensor,
        forces: torch.Tensor,
        state: Any,
    ) -> tuple[torch.Tensor, Any]:
        """
        One step of the rule from ``x``.

        Args:
            batch: the current batch
            x: current rows: the positions ``batch[self.key]``, with
                ``relax_cell`` followed by the cell rows
            forces: the force on every row, in ``x``'s dtype
            state: the history returned by the previous step, or by
                :meth:`init_state` for the first one

        Returns:
            The proposed rows and the history for the next step. The
            loop puts fixed atoms and converged structures back.
        """
        raise NotImplementedError


class GradientDescent(Optimizer):
    """
    Gradient descent, x <- x + eps F.

    With a physical force ``step_size`` is in Angstrom^2/eV; with a
    pseudo-force ``F = -grad |x - x0|^2`` it is unitless.
    """

    def __init__(
        self,
        calculator,
        step_size: float,
        fmax: float | None = None,
        structures: StructureProvider | None = None,
        prior: Prior | None = None,
        hooks: Sequence = (),
        key: str = properties.R,
        relax_cell: bool = False,
    ):
        """
        Args:
            calculator: see :class:`Optimizer`
            step_size: eps, the factor on the force
            fmax: see :class:`Optimizer`
            structures: see :class:`Optimizer`
            prior: see :class:`Optimizer`
            hooks: see :class:`Optimizer`
            key: see :class:`Optimizer`
            relax_cell: see :class:`Optimizer`
        """
        super().__init__(
            calculator,
            structures=structures,
            prior=prior,
            hooks=hooks,
            key=key,
            fmax=fmax,
            relax_cell=relax_cell,
        )
        self.step_size = step_size

    def step(self, batch, x, forces, state):
        return x + self.step_size * forces, state

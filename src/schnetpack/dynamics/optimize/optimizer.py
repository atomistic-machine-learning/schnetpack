"""
The time-free family: drivers that step structures along a force until they
are relaxed, or for a fixed number of steps.

:class:`Optimizer` owns the loop: the stop test, the holding of converged
structures and fixed atoms and the noise injection. A subclass is one step rule on the force —
:class:`Langevin` (gradient descent at kT = 0),
:class:`~schnetpack.dynamics.optimize.LBFGS`,
:class:`~schnetpack.dynamics.optimize.DirectDenoising`. The force comes from one
:class:`~schnetpack.dynamics.calculator.ForceCalculator`.
"""

import abc
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.calculator import Calculator, ForceCalculator
from schnetpack.dynamics.optimize.noise import ConstantNoise, NoiseSchedule
from schnetpack.generative.priors import Prior

__all__ = ["Optimizer", "Langevin"]


class Optimizer(Dynamics):
    """
    Base class of the time-free drivers: x <- step(x, F(x)), with optional
    noise before every step.

    Each step evaluates the force at the current structures and hands it to
    :meth:`step`, the subclass's rule. Before the force is evaluated, a
    ``noise`` schedule, if any, adds ``scale * z`` to the moved atoms. With an
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
    their forces are zeroed before the step rule sees them, they receive no
    noise, their step is zeroed after, and they are left out of the stop
    test.

    Constraints run around every step as in any
    :class:`~schnetpack.dynamics.base.Dynamics`, after the noise injection.
    A run that converges early stops before ``n_steps``, so a constraint's
    final ``after_step(step == n_steps)`` only fires when the step limit is
    reached.
    """

    time_free = True
    """No noise level: constraints overwrite rather than re-noise."""

    def __init__(
        self,
        calculator,
        prior: Prior | None = None,
        constraints: Sequence = (),
        key: str = properties.R,
        fmax: float | None = None,
        noise: NoiseSchedule | None = None,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.ForceCalculator`,
                or a bare callable batch -> outputs in eV and Angstrom
            prior: starting distribution :meth:`sample` draws from
            constraints: state constraints applied around every step, in
                order
            key: batch key this driver moves
            fmax: stop criterion on the largest force, in the force's unit;
                None runs all ``n_steps``
            noise: schedule of the noise injected before every step; None
                injects nothing
        """
        if not isinstance(calculator, Calculator):
            calculator = ForceCalculator(calculator)
        elif not isinstance(calculator, ForceCalculator):
            raise TypeError(
                f"{type(self).__name__} runs on a ForceCalculator, which names "
                "the model's units and converts them to eV and Angstrom; got a "
                f"{type(calculator).__name__}"
            )
        calculator.cache_last = True
        super().__init__(calculator, prior=prior, constraints=constraints, key=key)
        self.fmax = fmax
        self.noise = noise

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

    def _converged(self, batch: Mapping[str, Any]) -> torch.Tensor | None:
        """
        Which structures in ``batch`` are relaxed: the largest force on any of
        their free atoms is below ``fmax``.

        Returns:
            ``(n_structures,)`` boolean mask, or None without an ``fmax``.
        """
        if self.fmax is None:
            return None
        squared = self.forces(batch).pow(2).sum(-1)
        n_structures = batch[properties.n_atoms].shape[0]
        max_sq = torch.zeros(
            n_structures, dtype=squared.dtype, device=squared.device
        ).scatter_reduce(0, batch[properties.idx_m], squared, "amax", include_self=True)
        return max_sq < self.fmax**2

    @staticmethod
    def _moves(
        batch: Mapping[str, Any], converged: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Which atoms may move: not fixed, and in a structure not yet
        converged. None when every atom may.
        """
        moves = None
        if converged is not None:
            moves = (~converged)[batch[properties.idx_m]]
        fixed = batch.get(properties.fixed_atoms)
        if fixed is not None:
            moves = ~fixed if moves is None else moves & ~fixed
        return moves

    def _inject(
        self,
        batch: Mapping[str, Any],
        step: int,
        n_steps: int,
        converged: torch.Tensor | None,
    ) -> dict[str, Any]:
        """``batch`` with the noise of ``step`` added to the moving atoms."""
        scale = self.noise(step, n_steps) if self.noise is not None else 0.0
        if scale <= 0.0:
            return batch
        x = batch[self.key]
        x_new = x + scale * torch.randn_like(x)
        moves = self._moves(batch, converged)
        if moves is not None:
            x_new = torch.where(moves.unsqueeze(-1), x_new, x)
        return {**batch, self.key: x_new}

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
        state = self.init_state(batch, batch[self.key])

        for i in range(n_steps):
            converged = self._converged(batch)
            if converged is not None and bool(converged.all()):
                break

            batch = self._inject(batch, i, n_steps, converged)
            batch = self.before_step(batch, i, n_steps)
            x = batch[self.key]
            x_new, state = self.step(batch, x, self.forces(batch).to(x.dtype), state)
            moves = self._moves(batch, converged)
            if moves is not None:
                x_new = torch.where(moves.unsqueeze(-1), x_new, x)
            batch = {**batch, self.key: x_new}
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

    def init_state(self, batch: Mapping[str, Any], x: torch.Tensor) -> Any:
        """
        The history the step rule carries across the steps of one run.

        Called once at loop entry; the default is no history.

        Args:
            batch: the starting batch, already prepared by the calculator
            x: the starting positions
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
            batch: the current batch, for its structure layout
            x: current positions, ``batch[self.key]``
            forces: :meth:`forces` at ``x``, cast to ``x``'s dtype
            state: the history returned by the previous step, or by
                :meth:`init_state` for the first one

        Returns:
            The proposed positions and the history for the next step. The
            loop puts fixed atoms and converged structures back.
        """
        raise NotImplementedError


class Langevin(Optimizer):
    """
    Overdamped Langevin dynamics, x <- x + eps F + sqrt(2 eps kT) z; gradient
    descent at kT = 0.

    At kT > 0 the chain samples the Boltzmann density exp(-E/kT) up to the
    discretization error of ``step_size``; it has no minimum to stop at, so
    ``fmax`` is refused. The noise is injected before each step's force
    evaluation, so the returned batch is the last step's deterministic
    update.

    With a physical force ``step_size`` is in Angstrom^2/eV and ``kT`` in
    eV; with a pseudo-force F = -grad |x - x0|^2 ``step_size`` is unitless
    and ``kT`` in Angstrom^2.
    """

    def __init__(
        self,
        calculator,
        step_size: float,
        kT: float = 0.0,
        fmax: float | None = None,
        prior: Prior | None = None,
        constraints: Sequence = (),
        key: str = properties.R,
    ):
        """
        Args:
            calculator: see :class:`Optimizer`
            step_size: eps, the factor on the force
            kT: temperature; 0 is gradient descent
            fmax: stop criterion, kT = 0 only (see :class:`Optimizer`)
            prior: see :class:`Optimizer`
            constraints: see :class:`Optimizer`
            key: see :class:`Optimizer`
        """
        if kT < 0.0:
            raise ValueError(f"kT must be non-negative, got {kT}")
        if kT > 0.0 and fmax is not None:
            raise ValueError(
                "Langevin at kT > 0 samples; it has no minimum to stop at, so "
                "fmax must be None"
            )
        noise = ConstantNoise(math.sqrt(2.0 * step_size * kT)) if kT > 0.0 else None
        super().__init__(
            calculator,
            prior=prior,
            constraints=constraints,
            key=key,
            fmax=fmax,
            noise=noise,
        )
        self.step_size = step_size
        self.kT = kT

    def step(self, batch, x, forces, state):
        return x + self.step_size * forces, state

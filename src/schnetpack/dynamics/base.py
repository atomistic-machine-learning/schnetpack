"""
What every driver shares: the calculator, the starting distribution, the
moved key, the constraint hooks and the batch contract.

The structure is the batch dict used by datasets, transforms and models. A
driver moves one key and carries everything else along; inference goes
through a :class:`~schnetpack.dynamics.calculator.Calculator`, which works on
a copy, so derived keys never land in the driver's batch. Details:
``docs_new/sampling.md`` §7.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from schnetpack import properties
from schnetpack.dynamics.calculator import as_calculator
from schnetpack.dynamics.constraints.field import FieldConstraint
from schnetpack.dynamics.constraints.state import StateConstraint
from schnetpack.generative.priors import Prior

__all__ = ["Dynamics"]


class Dynamics(abc.ABC):
    """
    Base class of the loops that move structures with a model.

    Every driver writes its loop as

        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)
            batch = <one step>
            batch = self.after_step(batch, i + 1, n_steps)

    so state constraints run in the same order everywhere, between full steps
    only. Field constraints
    (:class:`~schnetpack.dynamics.constraints.field.FieldConstraint`) share
    the list but act inside the field, at every evaluation; the driver folds
    :meth:`constraint_field` into whatever its field is. :meth:`sample` draws
    starting structures from the prior and hands them to :meth:`denoise`,
    which also takes given structures.
    """

    def __init__(
        self,
        calculator,
        prior: Prior | None = None,
        constraints: Sequence = (),
        key: str = properties.R,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.Calculator`, or a bare
                callable batch -> outputs
            prior: starting distribution :meth:`sample` draws from; without
                one, only :meth:`denoise` on given structures is available
            constraints: state-level constraints applied around every step,
                in order, and field-level constraints added to the field
            key: batch key this driver moves
        """
        self.calculator = as_calculator(calculator)
        self.prior = prior
        self.constraints = list(constraints)
        for constraint in self.constraints:
            if not isinstance(constraint, (StateConstraint, FieldConstraint)):
                raise TypeError(
                    f"{type(constraint).__name__} is neither a StateConstraint "
                    "nor a FieldConstraint"
                )
        self.key = key

    @property
    def state_constraints(self) -> list[StateConstraint]:
        """The state constraints of :attr:`constraints`, in order."""
        return [c for c in self.constraints if isinstance(c, StateConstraint)]

    @property
    def field_constraints(self) -> list[FieldConstraint]:
        """The field constraints of :attr:`constraints`."""
        return [c for c in self.constraints if isinstance(c, FieldConstraint)]

    def before_step(self, batch: dict, step: int, n_steps: int) -> dict:
        """Run the state constraints' before-step hooks, in order."""
        for constraint in self.state_constraints:
            batch = constraint.before_step(batch, step, n_steps, self)
        return batch

    def after_step(self, batch: dict, step: int, n_steps: int) -> dict:
        """Run the state constraints' after-step hooks, in order."""
        for constraint in self.state_constraints:
            batch = constraint.after_step(batch, step, n_steps, self)
        return batch

    def constraint_field(
        self, batch: Mapping[str, Any], positions
    ) -> torch.Tensor | None:
        """
        The field the field constraints add at ``positions``: the sum of
        their terms, each scaled by its ``weight``.

        Every constraint is called on a copy of ``batch`` with ``positions``
        under ``properties.R``.

        Args:
            batch: current batch
            positions: positions in Angstrom to evaluate at

        Returns:
            The field ``(n_atoms, 3)`` in eV/Angstrom, or None without field
            constraints.
        """
        constraints = self.field_constraints
        if not constraints:
            return None
        inputs = {**batch, properties.R: positions}
        field = torch.zeros_like(positions)
        for constraint in constraints:
            field = field + constraint.weight * constraint(inputs)
        return field

    def sample(self, n_samples: int, n_steps: int) -> dict[str, Any]:
        """
        Draw ``n_samples`` starting structures from the prior and denoise.

        Args:
            n_samples: number of structures
            n_steps: number of steps

        Returns:
            The final batch.
        """
        if self.prior is None:
            raise ValueError(
                f"{type(self).__name__} has no prior to sample from; pass one "
                "at construction or call denoise on given structures"
            )
        return self.denoise(self.prior.sample(n_samples), n_steps)

    @abc.abstractmethod
    def denoise(self, batch: Mapping[str, Any], n_steps: int) -> dict[str, Any]:
        """
        Run the loop on the structures in ``batch``. Drivers may add keywords.

        Args:
            batch: structures to denoise
            n_steps: number of steps

        Returns:
            The final batch.
        """
        raise NotImplementedError

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

from schnetpack import properties
from schnetpack.dynamics.calculator import as_calculator
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
    only. :meth:`sample` draws starting structures from the prior and hands
    them to :meth:`denoise`, which also takes given structures.
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
                in order
            key: batch key this driver moves
        """
        self.calculator = as_calculator(calculator)
        self.prior = prior
        self.constraints = list(constraints)
        self.key = key

    def before_step(self, batch: dict, step: int, n_steps: int) -> dict:
        """Run the constraints' before-step hooks, in order."""
        for constraint in self.constraints:
            batch = constraint.before_step(batch, step, n_steps, self)
        return batch

    def after_step(self, batch: dict, step: int, n_steps: int) -> dict:
        """Run the constraints' after-step hooks, in order."""
        for constraint in self.constraints:
            batch = constraint.after_step(batch, step, n_steps, self)
        return batch

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

"""
What every driver shares: the calculator, the starting structures and
their prior, the moved key, the hooks and the batch contract.

The structure is the batch dict used by datasets, transforms and models. A
driver moves one key and carries everything else along; inference goes
through a :class:`~schnetpack.dynamics.calculator.Calculator`, which works on
a copy, so derived keys never land in the driver's batch. Details:
``docs_new/sampling.md`` §7.
"""

import abc
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from schnetpack import properties
from schnetpack.dynamics.calculator import Calculator
from schnetpack.dynamics.guidance import Guidance
from schnetpack.dynamics.hooks import Hook
from schnetpack.dynamics.structures import StructureProvider
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

    so hooks run in the same order everywhere, between full steps
    only. Terms that change the field itself
    (:class:`~schnetpack.dynamics.guidance.Guidance`) belong to the
    force calculator, which returns the guided force, or to a sampler's
    updates, which guide their own field. :meth:`sample` draws
    starting structures from the structure provider, redraws them from the
    prior, if any, and hands them to :meth:`run`, which also takes given
    structures.
    """

    def __init__(
        self,
        calculator: Calculator,
        structures: StructureProvider | None = None,
        prior: Prior | None = None,
        hooks: Sequence = (),
        key: str = properties.R,
    ):
        """
        Args:
            calculator: runs the model; each family asks for its own kind
                (:class:`~schnetpack.dynamics.calculator.ForceCalculator` or
                :class:`~schnetpack.dynamics.calculator.FieldCalculator`)
            structures: source of the structures :meth:`sample` starts
                from; without one, only :meth:`run` on given structures is
                available
            prior: redraws the structures' positions before :meth:`sample`
                runs; without one they are run as provided
            hooks: run around every step, in order
            key: batch key this driver moves
        """
        self.calculator = calculator
        self.structures = structures
        self.prior = prior
        self.hooks = list(hooks)
        for hook in self.hooks:
            if isinstance(hook, Guidance):
                raise TypeError(
                    f"{type(hook).__name__} is guidance, which changes "
                    "the field: pass it as guidance to the ForceCalculator, "
                    "or to an update of a Sampler"
                )
            if not isinstance(hook, Hook):
                raise TypeError(f"{type(hook).__name__} is not a Hook")
        self.key = key

    def before_step(self, batch: dict, step: int, n_steps: int) -> dict:
        """Run the hooks' ``before_step``, in order."""
        for hook in self.hooks:
            batch = hook.before_step(batch, step, n_steps)
        return batch

    def after_step(self, batch: dict, step: int, n_steps: int) -> dict:
        """Run the hooks' ``after_step``, in order."""
        for hook in self.hooks:
            batch = hook.after_step(batch, step, n_steps)
        return batch

    def sample(self, n_samples: int, n_steps: int) -> dict[str, Any]:
        """
        Draw ``n_samples`` starting structures and run on them.

        The structures come from :attr:`structures`; a :attr:`prior`
        redraws the keys it owns.

        Args:
            n_samples: number of structures
            n_steps: number of steps

        Returns:
            The final batch.
        """
        self._check_structures()
        return self.run(self._start_from(self.structures(n_samples)), n_steps)

    def sample_batches(
        self, n_samples: int, n_steps: int, batch_size: int
    ) -> Iterator[dict[str, Any]]:
        """
        Draw ``n_samples`` structures in batches and run on each, as successive
        :meth:`sample` calls of at most ``batch_size`` structures would.

        The structures of the next batch are drawn while the device still runs
        the current one: a run only queues its steps, and the caller waits for
        them when it reads the batch. Drawing on the cpu, e.g. reading a
        dataset, then costs no time on a GPU.

        Args:
            n_samples: number of structures in all
            n_steps: number of steps of every run
            batch_size: structures per run; the last run takes the rest

        Yields:
            The final batch of each run.
        """
        self._check_structures()
        sizes = [batch_size] * (n_samples // batch_size)
        if n_samples % batch_size:
            sizes.append(n_samples % batch_size)
        if not sizes:
            return
        structures = self.structures(sizes[0])
        for next_size in [*sizes[1:], None]:
            batch = self.run(self._start_from(structures), n_steps)
            if next_size is not None:
                structures = self.structures(next_size)
            yield batch

    def _check_structures(self) -> None:
        if self.structures is None:
            raise ValueError(
                f"{type(self).__name__} has no structures to sample from; pass "
                "structures=DatasetStructures(...) or StatisticsStructures(...) "
                "at construction, or call run on given structures"
            )

    def _start_from(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Where a run on drawn structures starts: the prior's draw, if any."""
        return batch if self.prior is None else self.prior(batch)

    @abc.abstractmethod
    def run(self, batch: Mapping[str, Any], n_steps: int) -> dict[str, Any]:
        """
        Run the loop on the structures in ``batch``. Drivers may add keywords.

        Args:
            batch: structures to move
            n_steps: number of steps

        Returns:
            The final batch.
        """
        raise NotImplementedError

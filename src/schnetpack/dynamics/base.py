"""
What every driver shares: the calculator, the starting distribution, the
moved key, the constraint hooks and the batch contract.

Sampling a generative model and relaxing a structure differ in what one step
does and when to stop. Each driver writes its own loop — a plain ``for``
over the steps — and calls :meth:`~Dynamics.before_step` /
:meth:`~Dynamics.after_step` around every step, so the constraint order is
the same in every driver.

:class:`Dynamics` knows nothing about generative models, so a force-field
optimizer (L-BFGS and friends) needs no process, parametrization or time
key. The drivers that run a generative model —
:class:`~schnetpack.dynamics.sampling.Sampler`,
:class:`~schnetpack.dynamics.relax.DirectDenoising` — hold those
themselves. Every driver has the same two entries: :meth:`Dynamics.sample`
draws starting structures from a prior (a generative model's noise
distribution, or noisy structures from a dataset) and hands them to
:meth:`Dynamics.denoise`, the loop, which also takes given structures.

The structure is the batch dict — the same one datasets, transforms and
models use — and nothing else. A driver moves one key (:attr:`Dynamics.key`),
reads it out of the batch for its pure-tensor math and writes the result
back into a new dict. Inference — device, neighbor list, gradient policy,
the model call — goes through a
:class:`~schnetpack.dynamics.calculator.Calculator`, which works on a copy,
so the keys it computes never land in the driver's batch.
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

    Holds the calculator that runs the model, the starting distribution, the
    constraints and the batch-dict contract:

    - the model is reached through ``self.calculator``, a
      :class:`~schnetpack.dynamics.calculator.Calculator` (a bare
      ``batch -> outputs`` callable is wrapped in one), called with a batch
      and nothing else.
    - ``key`` is the batch key the driver moves; everything else in the
      batch is carried along untouched.

    :meth:`sample` draws starting structures from the prior and hands them to
    :meth:`denoise`, the loop, which every driver writes out as

        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)       # constraints, in order
            batch = <one step>
            batch = self.after_step(batch, i + 1, n_steps)    # constraints, in order

    Constraints (:class:`~schnetpack.dynamics.constraints.state.StateConstraint`)
    act between full steps only — never between the stages of a multi-stage
    integrator such as Heun. Their order is the list order, and it matters:
    a constraint that overwrites atoms should run after one that perturbs
    them.
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
        Run the loop on the structures in ``batch``: the entry point for
        given structures — relaxation, scaffolded generation, partial
        denoising. Drivers may add keywords (the sampler's ``t_start``).

        Args:
            batch: structures to denoise
            n_steps: number of steps

        Returns:
            The final batch.
        """
        raise NotImplementedError

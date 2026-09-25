"""
What every driver shares: the trained pair, the starting distribution, the
constraints and the batch contract.

Sampling a generative model and relaxing a structure differ in what one step
does and when to stop. Each driver writes its own loop in :meth:`~Dynamics.denoise`
— a plain ``for`` over the steps — and calls :meth:`~Dynamics.before_step` /
:meth:`~Dynamics.after_step` around every step, so the constraint order is
the same in every driver.

The structure is the batch dict — the same one datasets, transforms and
models use — and nothing else. A driver moves one key (:attr:`Dynamics.key`),
reads it out of the batch for the pure-tensor process/parametrization/
integrator math and writes the result back into a new dict. Inference —
device, neighbor list, gradient policy, the model call — goes through a
:class:`~schnetpack.dynamics.calculator.Calculator`, which works on a copy,
so the keys it computes never land in the driver's batch.
"""

import abc
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from schnetpack import properties
from schnetpack.dynamics.calculator import as_calculator
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process

__all__ = ["Dynamics"]


class Dynamics(abc.ABC):
    """
    Base class of the loops that move structures with a model.

    Holds the calculator that runs the model, the ``(process,
    parametrization)`` pair the model was trained under (validated at
    construction), the starting distribution, the constraints and the
    batch-dict contract:

    - the model is reached through ``self.calculator``, a
      :class:`~schnetpack.dynamics.calculator.Calculator` (a bare
      ``batch -> outputs`` callable is wrapped in one), called with a batch
      and nothing else; the driver reads its raw head from
      ``outputs[output_key]``. Time and conditioning reach the
      model as batch keys (``time_key``; conditioning keys are simply left in
      the batch).
    - ``key`` is the batch key the driver moves; everything else in the
      batch is carried along untouched.

    :meth:`sample` draws the moved key from the prior and hands the batch to
    :meth:`denoise`, the loop, which every driver writes out as

        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)       # constraints, in order
            batch = <one step>
            batch = self.after_step(batch, i + 1, n_steps)    # constraints, in order

    Constraints (:class:`~schnetpack.dynamics.constraints.StateConstraint`)
    act between full steps only — never between the stages of a multi-stage
    integrator such as Heun. Their order is the list order, and it matters:
    a constraint that overwrites atoms should run after one that perturbs
    them.
    """

    time_free: bool = False
    """Whether the loop runs without a noise level (the time key is all zeros).

    Constraints read this to decide how to act: a scaffold is re-noised to
    the current time on a time-aware path and simply overwritten on a
    time-free one.
    """

    def __init__(
        self,
        calculator,
        process: Process,
        parametrization: Parametrization,
        prior: Optional[Prior] = None,
        constraints: Sequence = (),
        key: str = properties.R,
        output_key: str = "prediction",
        time_key: str = properties.t,
    ):
        """
        Args:
            calculator: runs the model: a
                :class:`~schnetpack.dynamics.calculator.Calculator`, or a bare
                callable batch -> outputs
            process: forward process the model was trained on; supplies the
                sampling prior
            parametrization: contract the model was trained under
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal, where there is no data-free start to derive.
            constraints: state-level constraints applied around every step,
                in order
            key: batch key this driver moves
            output_key: model output holding the raw head, in the
                parametrization
            time_key: batch key the path time is written to, one value per
                row of the moved key — the key
                :class:`~schnetpack.generative.transforms.Diffuse` wrote in
                training
        """
        parametrization.validate(process)
        self.calculator = as_calculator(calculator)
        self.process = process
        self.parametrization = parametrization
        self.prior = prior if prior is not None else process.sampling_prior()
        self.constraints = list(constraints)
        self.key = key
        self.output_key = output_key
        self.time_key = time_key

    def before_step(self, batch: Dict, step: int, n_steps: int) -> Dict:
        """Run the constraints' before-step hooks, in order."""
        for constraint in self.constraints:
            batch = constraint.before_step(batch, step, n_steps, self)
        return batch

    def after_step(self, batch: Dict, step: int, n_steps: int) -> Dict:
        """Run the constraints' after-step hooks, in order."""
        for constraint in self.constraints:
            batch = constraint.after_step(batch, step, n_steps, self)
        return batch

    def sample(
        self,
        batch: Mapping[str, Any],
        n_steps: int,
    ) -> Dict[str, Any]:
        """
        Draw the moved key from the prior and denoise.

        Args:
            batch: template of the structures to generate: atom types,
                ``n_atoms``, ``idx_m`` and any conditioning. Values of the
                moved key are replaced by the prior draw; only their shape is
                used, and for positions it is derived from the atom types
                when absent. The prior reads the layout (``idx_m``) from here
                to center each structure on its own.
            n_steps: number of steps

        Returns:
            The final batch.
        """
        batch = self.calculator.prepare(batch)
        key = self.key
        if key in batch:
            like = batch[key]
            shape, dtype, device = like.shape, like.dtype, like.device
        elif key == properties.R and properties.Z in batch:
            z = batch[properties.Z]
            shape, dtype, device = (z.shape[0], 3), self.calculator.dtype, z.device
        else:
            raise KeyError(
                f"cannot infer the shape of {key!r}: put a placeholder of the "
                "right shape into the template batch"
            )
        x = self.prior.sample(shape, dtype=dtype, device=device, context=batch)
        return self.denoise({**batch, key: x}, n_steps)

    @abc.abstractmethod
    def denoise(
        self,
        batch: Mapping[str, Any],
        n_steps: int,
        t_start: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Denoise the structures in ``batch`` — the partial-denoising entry point.

        Relaxation of given structures, scaffolded generation and structured
        priors that start below t_max all enter here.

        Args:
            batch: structures to denoise
            n_steps: number of steps
            t_start: path time the structures are assumed to sit at; each
                subclass states its default and whether it reads it

        Returns:
            The final batch.
        """
        raise NotImplementedError

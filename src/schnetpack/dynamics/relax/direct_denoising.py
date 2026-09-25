"""
GPFF's direct denoising — a relaxation, not a time-stepped sampler.

The loop repeats "inject noise, jump to the model's x0-estimate". There is
no time grid, no reverse SDE/ODE and no noise schedule: the only ingredients
are ``parametrization.to_x0`` and the injection. That makes it a *relaxer* —
the jump x <- x + F/2 is the exact Newton step on the quadratic pseudo-energy
||x - x0||^2 the pseudo-force is the gradient of — which is why it lives here
and not with the grid-walking :class:`~schnetpack.dynamics.sampling.Sampler`.

The model contract is the batch dict (``batch -> outputs``, see
:class:`~schnetpack.dynamics.base.Dynamics`); the batch-dict relaxers
(L-BFGS and friends) arrive with the batch-wise optimizer port, and this loop
becomes one step rule among theirs.
"""

from collections.abc import Sequence

import torch

from schnetpack import properties
from schnetpack.dynamics.base import Dynamics
from schnetpack.dynamics.constraints.state import AnnealedNoise
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process

__all__ = ["DirectDenoising"]


class DirectDenoising(Dynamics):
    """
    GPFF's direct denoising: repeat "inject noise, jump to the model's
    x0-estimate".

    Each of the ``n_steps`` iterations does

        x <- x + lambda (1 - k/N) z,  z ~ N(0, I)   (decaying noise injection)
        x <- x0_hat(x)                              (jump to the x0-estimate)

    There is no time grid, no reverse SDE/ODE and no noise schedule, which is
    why this is a relaxer rather than an integrator of
    :class:`~schnetpack.dynamics.sampling.Sampler`: the
    only ingredients are ``parametrization.to_x0`` and the injection above.
    The step is the bare jump; the injection is an
    :class:`~schnetpack.dynamics.constraints.state.AnnealedNoise` constraint that
    ``stochastic_lambda`` puts first in the constraint list, so user
    constraints (a :class:`~schnetpack.dynamics.constraints.state.Scaffold`) act on
    the noised state. ``stochastic_lambda = 0`` disables the injection
    entirely (GPFF's plain direct denoising); positive values give the
    stochastic variant, whose injected noise is what buys sample diversity.
    lambda is in data units (Angstrom, for positions).

    The model is evaluated at t = 0 throughout — the sampler never knows the
    noise level of its iterate, so it presumes the *time-free* contract that
    makes GPFF's method possible in the first place: a model that ignores its
    t input, under a parametrization whose ``to_x0`` never reads t either
    (the pseudo-force and x0 heads; a score-type head divides by sigma(t) and
    would read the lie). Time-conditioned models belong in
    :class:`~schnetpack.dynamics.sampling.Sampler`.
    """

    time_free = True
    """The model is run at t = 0 throughout, so constraints such as
    :class:`~schnetpack.dynamics.constraints.state.Scaffold` overwrite
    rather than re-noise.
    """

    def __init__(
        self,
        calculator,
        process: Process,
        parametrization: Parametrization,
        prior: Prior | None = None,
        stochastic_lambda: float = 1.0,
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
            parametrization: contract the model was trained under; its
                ``to_x0`` is the jump
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal.
            stochastic_lambda: scale of the injected noise, in data units;
                0 disables the injection
            constraints: state-level constraints, applied after the noise
                injection
            key: batch key this driver moves
            output_key: model output holding the raw head, in the
                parametrization
            time_key: batch key the zero time is written to, for models
                that take a time input
        """
        parametrization.validate(process)
        injection = (
            [AnnealedNoise(stochastic_lambda)] if stochastic_lambda > 0.0 else []
        )
        super().__init__(
            calculator,
            prior=prior if prior is not None else process.sampling_prior(),
            constraints=injection + list(constraints),
            key=key,
        )
        self.process = process
        self.parametrization = parametrization
        self.output_key = output_key
        self.time_key = time_key
        self.stochastic_lambda = stochastic_lambda

    def denoise(self, batch, n_steps: int, t_start=None):
        """
        Relax the structures in ``batch`` by ``n_steps`` jumps.

        No ``t_start`` to declare, unlike
        :meth:`~schnetpack.dynamics.sampling.Sampler.denoise`: the loop never
        uses the noise level, which is exactly what makes relaxing structures
        of unknown noisiness this relaxer's home turf.

        Args:
            batch: structures to relax
            n_steps: number of jumps

        Returns:
            The final batch.
        """
        if t_start is not None:
            raise ValueError(
                "DirectDenoising is time-free and takes no t_start; "
                "use Sampler.denoise to start from a known noise level."
            )
        self.calculator.reset()
        batch = self.calculator.prepare(batch)
        x = batch[self.key]
        t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        batch = {**batch, self.time_key: t}
        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)
            raw = self.calculator(batch)[self.output_key]
            x = self.parametrization.to_x0(self.process, raw, batch[self.key], t)
            batch = {**batch, self.key: x}
            batch = self.after_step(batch, i + 1, n_steps)
        return batch

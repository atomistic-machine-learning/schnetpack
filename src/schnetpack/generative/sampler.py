"""
Composition of process, parametrization, integrator, grid and prior into a
sampler.
"""

from typing import Callable, Optional, Sequence

import torch

from schnetpack.generative.grids import TimeGrid, UniformGrid
from schnetpack.generative.integrators.base import Integrator
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.priors import Prior
from schnetpack.generative.processes import Process
from schnetpack.generative.reverse import ReverseProcess

__all__ = ["Sampler"]


class Sampler:
    """
    Thin composition wrapper: prior -> reverse process -> integrator.

    Takes the ``(process, parametrization)`` pair — the same pair the model
    was trained under; keeping the two sides consistent is the caller's job,
    so share the objects with the training code rather than rebuilding them.
    The pairing is checked at construction via
    :meth:`~schnetpack.generative.parametrizations.Parametrization.validate`.
    The starting distribution is not asked for by default — it *is* the
    process's sampling prior (the training prior itself, since b(t_max) = 1
    and the coupling preserves the marginal), so deriving it beats restating
    it. An explicit ``prior`` overrides that, and is required when the
    process cannot state its own start (a marginal-changing coupling).

    Operates at tensor level. ``model`` is any callable
    ``(x, t, cond) -> raw output`` in the parametrization; the sample axis is
    simply ``x.shape[0]``, whatever that means for your data. The adapter that
    wraps a SchNetPack :class:`~schnetpack.model.NeuralNetworkPotential` into
    that contract (where the sample axis is atoms) arrives with the atomistic
    port; this class stays unaware of it.

    Method-specific behavior belongs in the composed parts. If you find
    yourself subclassing this, the logic probably belongs in a process,
    parametrization, integrator or grid — that is what the axes are for.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        integrator: Integrator,
        grid: Optional[TimeGrid] = None,
        prior: Optional[Prior] = None,
        churn: float = 1.0,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ):
        """
        Args:
            process: forward process the model was trained on; supplies the
                schedule and the training prior
            parametrization: contract the model was trained under
            integrator: numerical solver for the reverse process
            grid: where to place the steps (default: uniform)
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal, where there is no data-free start to derive.
            churn: stochasticity of the reverse process; 1 = reverse SDE,
                0 = probability-flow ODE. Equals eta^2 of the Anderson family.
            t_min: time to stop integration at (default: ``process.t_min``);
                the score diverges as b -> 0
            t_max: time to start integration from (default: ``process.t_max``)
        """
        parametrization.validate(process)
        self.process = process
        self.parametrization = parametrization
        self.integrator = integrator
        self.grid = grid if grid is not None else UniformGrid()
        self.prior = prior if prior is not None else process.sampling_prior()
        self.churn = churn
        self.t_min = t_min if t_min is not None else process.t_min
        self.t_max = t_max if t_max is not None else process.t_max

    def sample(
        self,
        model: Callable,
        shape: Sequence[int],
        n_steps: int,
        x_init: Optional[torch.Tensor] = None,
        cond=None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Draw samples by integrating the reverse process from t_max down to t_min.

        Args:
            model: callable (x, t, cond) -> raw output in the sampler's
                parametrization
            shape: shape of the sample batch, (n_samples, ...)
            n_steps: number of integrator steps
            x_init: optional starting states; drawn from the prior if not given
            cond: conditioning passed through to the model
        """
        if x_init is None:
            x_init = self.prior.sample(shape, dtype=dtype, device=device)
        return self.denoise(model, x_init, self.t_max, n_steps, cond=cond)

    def denoise(
        self,
        model: Callable,
        x_t: torch.Tensor,
        t_start: float,
        n_steps: int,
        cond=None,
    ) -> torch.Tensor:
        """
        Denoise given samples from t_start down to t_min.

        This is the partial-denoising entry point: relaxation of given
        structures, scaffolded generation and structured priors that start
        below t_max all enter here.

        Args:
            model: callable (x, t, cond) -> raw output in the sampler's
                parametrization
            x_t: states to denoise, shape (n_samples, ...)
            t_start: path time the states are assumed to sit at
            n_steps: number of integrator steps
            cond: conditioning passed through to the model
        """
        ts = self.grid(t_start, self.t_min, n_steps, dtype=x_t.dtype, device=x_t.device)
        reverse = ReverseProcess(
            self.process, self.parametrization, model, churn=self.churn, cond=cond
        )
        return self.integrator.integrate(reverse, x_t, ts)

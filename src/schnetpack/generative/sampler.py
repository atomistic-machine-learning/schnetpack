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

__all__ = ["DirectDenoisingSampler", "Sampler"]


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
        context=None,
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
            context: generation-time conditioning handed to the prior — the
                same argument the forward process passes it during training,
                so a prior that needs more than a shape is reachable from
                here: composition, scaffold indices, or the batch layout a
                :class:`~schnetpack.generative.priors.GaussianPrior` centers
                by. Pass the batch dict to generate a batch of molecules, or
                each draw is centered against the whole cloud instead of its
                own structure. Ignored when ``x_init`` is given, and by priors
                that do not read it.
        """
        if x_init is None:
            x_init = self.prior.sample(
                shape, dtype=dtype, device=device, context=context
            )
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


class DirectDenoisingSampler:
    """
    GPFF's direct denoising: repeat "inject noise, jump to the model's
    x0-estimate".

    Each of the ``n_steps`` iterations does

        x <- x + lambda (1 - k/N) z,  z ~ N(0, I)   (decaying noise injection)
        x <- x0_hat(x)                              (jump to the x0-estimate)

    There is no time grid, no reverse SDE/ODE and no noise schedule, which is
    why this is a sibling of :class:`Sampler` rather than an integrator: the
    only ingredients are ``parametrization.to_x0`` and the injection above.
    ``stochastic_lambda = 0`` disables the injection entirely (GPFF's plain
    direct denoising); positive values give the stochastic variant, whose
    injected noise is what buys sample diversity. lambda is in data units
    (Angstrom, for positions).

    The model is evaluated at t = 0 throughout — the sampler never knows the
    noise level of its iterate, so it presumes the *time-free* contract that
    makes GPFF's method possible in the first place: a model that ignores its
    t argument, under a parametrization whose ``to_x0`` never reads t either
    (the pseudo-force and x0 heads; a score-type head divides by sigma(t) and
    would read the lie). Time-conditioned models belong in :class:`Sampler`.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        prior: Optional[Prior] = None,
        stochastic_lambda: float = 1.0,
    ):
        """
        Args:
            process: forward process the model was trained on; supplies the
                sampling prior
            parametrization: contract the model was trained under; its
                ``to_x0`` is the jump
            prior: explicit starting distribution; overrides the process's
                own. Required when the process's coupling changes x1's
                marginal.
            stochastic_lambda: scale of the injected noise, in data units;
                0 disables the injection
        """
        parametrization.validate(process)
        self.process = process
        self.parametrization = parametrization
        self.prior = prior if prior is not None else process.sampling_prior()
        self.stochastic_lambda = stochastic_lambda

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
        Draw samples by direct denoising from the prior.

        Args:
            model: callable (x, t, cond) -> raw output in the sampler's
                parametrization; called with t = 0, so it must be time-free
            shape: shape of the sample batch, (n_samples, ...)
            n_steps: number of denoising iterations
            x_init: optional starting states; drawn from the prior if not given
            cond: conditioning passed through to the model
        """
        if x_init is None:
            x_init = self.prior.sample(shape, dtype=dtype, device=device)
        return self.denoise(model, x_init, n_steps, cond=cond)

    def denoise(
        self,
        model: Callable,
        x_t: torch.Tensor,
        n_steps: int,
        cond=None,
    ) -> torch.Tensor:
        """
        Denoise given states — the partial-denoising entry point.

        Unlike :meth:`Sampler.denoise` there is no ``t_start`` to declare:
        the loop never uses the noise level, which is exactly what makes
        relaxing structures of unknown noisiness this sampler's home turf.

        Args:
            model: callable (x, t, cond) -> raw output in the sampler's
                parametrization; called with t = 0, so it must be time-free
            x_t: states to denoise, shape (n_samples, ...)
            n_steps: number of denoising iterations
            cond: conditioning passed through to the model
        """
        x = x_t
        t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for k in range(1, n_steps + 1):
            noise_scale = self.stochastic_lambda * (1.0 - k / n_steps)
            if noise_scale > 0.0:
                x = x + noise_scale * torch.randn_like(x)
            raw = model(x, t, cond)
            x = self.parametrization.to_x0(self.process, raw, x, t)
        return x

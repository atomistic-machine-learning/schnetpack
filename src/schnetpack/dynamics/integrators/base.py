"""
Numerical steppers shared by the drivers: solvers for (reverse-time)
diffusion processes and step rules for relaxation.

An integrator consumes only ``dynamics.drift`` and ``dynamics.diffusion`` — it
is agnostic to what it integrates: a reverse SDE, a probability-flow ODE (g = 0),
a forward process, or the force field of a relaxation (drift = forces,
diffusion = 0, where Euler is steepest descent). Denoising runs backwards in
time, so ``dt < 0``.

Instances are stateless. An integrator that needs a history across steps — a
quasi-Newton step rule such as :class:`~schnetpack.dynamics.integrators.LBFGS`,
or a multistep solver — keeps it in a per-run ``state``: the driver asks
:meth:`Integrator.init_state` for it at loop entry and threads it through
every :meth:`Integrator.step`, which returns the next one. The one-step
solvers carry ``None``.
"""

import abc
from typing import Any

import torch

__all__ = ["Integrator"]


class Integrator(abc.ABC):
    """Base class for SDE/ODE solvers."""

    requires_sde: bool = False
    """Whether this integrator steps through the (f, g) chart's closed forms.

    False for the generic solvers, which consume only ``drift`` and
    ``diffusion`` and work on any reverse process. True for the ones that
    discretize through the chart itself — the ancestral steps, which read
    the exact posterior or the raw score — so the
    :class:`~schnetpack.dynamics.sampling.sampler.Sampler` can demand a
    :class:`~schnetpack.generative.differential_equations.ReverseSDE` at assembly instead of
    failing mid-run.
    """

    requires_structure: bool = False
    """Whether this integrator works per structure rather than per row.

    False for the solvers whose update acts on every row of x on its own.
    True for the step rules that reduce over each structure of the batch —
    L-BFGS's dot products and per-structure step length — and so read the
    structure layout (``idx_m``, ``n_atoms``) off the field they are handed. Only a
    :class:`~schnetpack.dynamics.relax.Relaxer` provides that field; the
    :class:`~schnetpack.dynamics.sampling.sampler.Sampler` refuses such an
    integrator at assembly instead of failing mid-run.
    """

    def init_state(self, dynamics, x: torch.Tensor) -> Any:
        """
        The history this integrator carries across the steps of one run.

        Called once at loop entry; the default is no history.

        Args:
            dynamics: the field the run integrates (see :meth:`step`)
            x: starting state
        """
        return None

    @abc.abstractmethod
    def step(
        self,
        dynamics,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        state: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """
        Advance x from t to t + dt.

        Args:
            dynamics: object exposing drift(x, t) and diffusion(t)
            x: current state, shape (batch, ...)
            t: current time, shape (batch,)
            dt: time increment (0-dim tensor; negative when denoising)
            state: the history returned by the previous step, or by
                :meth:`init_state` for the first one

        Returns:
            The advanced state and the history for the next step.
        """
        raise NotImplementedError

"""
Integrators: the step rule an SDE or ODE update applies along the time grid.

An integrator is an explicit stage scheme on a drift and never evaluates the
model. The sampler runs its stages: for each stage the update asks
:meth:`Integrator.point` where its variable goes next, given the drifts of
the stages so far, and finally hands all stage drifts to
:meth:`Integrator.combine` for the new state. :attr:`Integrator.nodes` says
where the stages sit in time, so the sampler can call the model once per
stage for all variables. The noise of an SDE step is the update's, drawn by
the variable's prior; an integrator adds none. So :class:`Euler` and
:class:`Heun` serve the
:class:`~schnetpack.dynamics.sample.sampler.SDEUpdate` and the
:class:`~schnetpack.dynamics.sample.sampler.ODEUpdate` alike. Ancestral
sampling is not an integrator but its own
:class:`~schnetpack.dynamics.sample.sampler.AncestralUpdate`.
"""

import abc

import torch

from schnetpack.generative.noising import expand_t

__all__ = ["Integrator", "Euler", "Heun"]


class Integrator(abc.ABC):
    """An explicit stage scheme advancing x along a drift from t to t_next."""

    n_stages: int
    """Number of drift evaluations per step."""

    nodes: tuple[float, ...]
    """Time of each stage, as a fraction of the step: 0 is t, 1 is t_next."""

    @abc.abstractmethod
    def point(
        self,
        stage: int,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        drifts: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Where to evaluate the drift for ``stage``.

        Args:
            stage: stage index, 0 <= stage < n_stages
            x: state at the start of the step, shape (n_rows, ...)
            t: time at the start of the step, 0-dim or shape (n_rows,)
            t_next: time at the end of the step, shaped like t
            drifts: the drifts of the stages before ``stage``

        Returns:
            (x_stage, t_stage): state and time to evaluate the drift at.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def combine(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        drifts: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        The deterministic state at ``t_next`` from all stage drifts.

        Args:
            x: state at the start of the step, shape (n_rows, ...)
            t: time at the start of the step, 0-dim or shape (n_rows,)
            t_next: time at the end of the step, shaped like t
            drifts: the drifts of all n_stages stages

        Returns:
            The advanced state, before any noise.
        """
        raise NotImplementedError


class Euler(Integrator):
    """
    One stage: x <- x + d0 dt. With the update's noise g sqrt(|dt|) z this
    is Euler–Maruyama.
    """

    n_stages = 1
    nodes = (0.0,)

    def point(self, stage, x, t, t_next, drifts):
        return x, t

    def combine(self, x, t, t_next, drifts):
        return x + drifts[0] * expand_t(t_next - t, x)


class Heun(Integrator):
    """
    Two stages: the predictor x + d0 dt at t_next, then
    x <- x + 1/2 (d0 + d1) dt. The update adds one Euler–Maruyama noise
    increment. Without noise this is the EDM (Karras et al. 2022) sampler.
    """

    n_stages = 2
    nodes = (0.0, 1.0)

    def point(self, stage, x, t, t_next, drifts):
        if stage == 0:
            return x, t
        return x + drifts[0] * expand_t(t_next - t, x), t_next

    def combine(self, x, t, t_next, drifts):
        return x + 0.5 * (drifts[0] + drifts[1]) * expand_t(t_next - t, x)

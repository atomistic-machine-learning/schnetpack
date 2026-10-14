"""
Integrators: the step rule a sampler applies along its time grid.

An integrator is an explicit stage scheme and never evaluates the model.
The sampler runs its stages: for each stage it asks :meth:`Integrator.point`
where to evaluate next, given the drifts of the stages so far, evaluates the
drift there, and finally hands all stage drifts and the diffusion to
:meth:`Integrator.combine` for the new state. So :class:`Euler` and
:class:`Heun` serve the
:class:`~schnetpack.dynamics.sample.sampler.SDESampler` and the
:class:`~schnetpack.dynamics.sample.sampler.ODESampler` alike. Ancestral
sampling is not an integrator but its own
:class:`~schnetpack.dynamics.sample.sampler.AncestralSampler`.
"""

import abc

import torch

from schnetpack.generative.noising import expand_t

__all__ = ["Integrator", "Euler", "Heun"]


class Integrator(abc.ABC):
    """An explicit stage scheme advancing x from t to t_next."""

    n_stages: int
    """Number of drift evaluations per step."""

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
            t: time at the start of the step, shape (n_rows,)
            t_next: time at the end of the step, shape (n_rows,)
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
        diffusion: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        The state at ``t_next`` from all stage drifts.

        Args:
            x: state at the start of the step, shape (n_rows, ...)
            t: time at the start of the step, shape (n_rows,)
            t_next: time at the end of the step, shape (n_rows,)
            drifts: the drifts of all n_stages stages
            diffusion: diffusion g(t) at the start of the step, shaped like
                t, or None for a deterministic step

        Returns:
            The advanced state.
        """
        raise NotImplementedError


class Euler(Integrator):
    """
    One stage: x <- x + d0 dt + g sqrt(|dt|) z (Euler–Maruyama); plain Euler
    without diffusion.
    """

    n_stages = 1

    def point(self, stage, x, t, t_next, drifts):
        return x, t

    def combine(self, x, t, t_next, drifts, diffusion):
        x_new = x + drifts[0] * expand_t(t_next - t, x)
        if diffusion is None:
            return x_new
        scale = diffusion * (t_next - t).abs().sqrt()
        return x_new + expand_t(scale, x) * torch.randn_like(x)


class Heun(Integrator):
    """
    Two stages: the predictor x + d0 dt at t_next, then
    x <- x + 1/2 (d0 + d1) dt, plus one Euler–Maruyama diffusion increment.
    Without diffusion this is the EDM (Karras et al. 2022) sampler.
    """

    n_stages = 2

    def point(self, stage, x, t, t_next, drifts):
        if stage == 0:
            return x, t
        return x + drifts[0] * expand_t(t_next - t, x), t_next

    def combine(self, x, t, t_next, drifts, diffusion):
        x_new = x + 0.5 * (drifts[0] + drifts[1]) * expand_t(t_next - t, x)
        if diffusion is None:
            return x_new
        scale = diffusion * (t_next - t).abs().sqrt()
        return x_new + expand_t(scale, x) * torch.randn_like(x)

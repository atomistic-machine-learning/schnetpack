"""
Matching losses: score, flow and pseudo-force matching as one training step.

Every objective is the same three moves (draw an endpoint pair, place it on
the path at a random time, regress the parametrization's target); which
process and parametrization are handed over decides the method. Tensor
level: :class:`MatchingLoss` takes a callable ``model(x, t, cond)`` and a
batch of samples, not a SchNetPack batch dict. For the data-pipeline route
see :class:`~schnetpack.generative.transforms.Diffuse`. Details:
``docs_new/training.md``.
"""

from collections.abc import Callable

import torch

from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process, expand_t

__all__ = ["MatchingLoss"]


class MatchingLoss:
    """
    Weighted regression of a parametrization's target along a process:
    ``mean(w(t) (model(x_t, t) - target)^2)``.

    The (process, parametrization) pair is validated at construction.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        weight: Callable[[torch.Tensor], torch.Tensor] | None = None,
        t_sampler: Callable[[int, torch.device], torch.Tensor] | None = None,
    ):
        """
        Args:
            process: forward process that draws and places the endpoints
            parametrization: what the model predicts, hence what to regress
            weight: per-sample loss weight w(t), mapping (n_samples,) ->
                (n_samples,) (default: uniform)
            t_sampler: draws training times, mapping (n_samples, device) ->
                (n_samples,) (default: the process's own
                :meth:`~schnetpack.generative.processes.Process.sample_t`);
                see :mod:`schnetpack.generative.times`
        """
        parametrization.validate(process)
        self.process = process
        self.parametrization = parametrization
        self.weight = weight if weight is not None else (lambda t: torch.ones_like(t))
        self.t_sampler = t_sampler

    def __call__(
        self,
        model: Callable,
        x0: torch.Tensor,
        x1: torch.Tensor | None = None,
        cond=None,
        context=None,
    ) -> torch.Tensor:
        """
        Args:
            model: callable (x, t, cond) -> raw output in ``parametrization``
            x0: clean samples, shape (n_samples, ...)
            x1: endpoints to use instead of drawing from the process's prior;
                still passed through its coupling
            cond: conditioning passed through to the model
            context: batch handed to the prior when drawing x1

        Returns:
            Scalar loss.
        """
        t = None
        if self.t_sampler is not None:
            t = self.t_sampler(x0.shape[0], x0.device).to(x0.dtype)
        x_t, x0, x1, t, eps = self.process.perturb(x0, x1, t=t, context=context)

        prediction = model(x_t, t, cond)
        target = self.parametrization.target(self.process, x0, x1, t, eps)
        w = expand_t(self.weight(t), x_t)
        return (w * (prediction - target) ** 2).mean()

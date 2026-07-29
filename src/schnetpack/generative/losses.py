"""
Matching losses — score, flow and bridge matching are one training step.

Every objective in this subpackage is the same three moves: draw an endpoint
pair, place it on the path at a random time, and regress the
parametrization's target. What distinguishes score matching from flow
matching from bridge matching is which process and parametrization you hand
it — not which loss you call. The three moves themselves belong to the
process (:meth:`~schnetpack.generative.processes.Process.perturb`);
this class adds the model call, the weighting and the MSE.

The ``(process, parametrization)`` pair is validated at construction: the
score/noise parametrizations demand the Gaussian kernel, which the process
judges from its own configuration
(:meth:`~schnetpack.generative.processes.Process.gaussian_kernel_obstruction`).
By the time a MatchingLoss exists, the assembly is coherent.

Pure PyTorch, tensor level. These take a model and a batch of samples, not a
SchNetPack batch dict; the adapter that maps atomistic batches onto this
contract, and the :class:`~schnetpack.objectives.UnsupervisedModelOutput` that
carries the result into the Lightning task, arrive with the atomistic port.
"""

from typing import Callable, Optional

import torch

from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process, expand_t

__all__ = ["MatchingLoss"]


class MatchingLoss:
    """
    Weighted regression of a parametrization's target along an interpolant path.

    Computes ``mean(w(t) (model(x_t, t) - target)^2)`` for x_t drawn from the
    process at times from ``t_sampler``.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        weight: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        t_sampler: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
    ):
        """
        Args:
            process: forward process that draws and places the endpoints
            parametrization: what the model predicts, and hence what to
                regress
            weight: per-sample loss weight w(t), mapping (n_samples,) ->
                (n_samples,) (default: uniform)
            t_sampler: draws training times, mapping (n_samples, device) ->
                (n_samples,). Defaults to the process's own
                :meth:`~schnetpack.generative.processes.Process.sample_t`
                — uniform on [t_min, t_max], stopping short of t = 0 because
                the score target diverges there. The noise and denoiser
                targets are well behaved at 0, so for those on a VP path you
                may widen the range back to [0, t_max]; the EDM/GPFF
                log-normal-sigma density enters through the same hook.
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
        x1: Optional[torch.Tensor] = None,
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
            context: generation-time conditioning handed to the prior

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

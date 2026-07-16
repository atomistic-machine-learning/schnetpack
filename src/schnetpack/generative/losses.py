"""
Matching losses — score, flow and bridge matching are one training step.

Every objective in this subpackage is the same three moves: draw an endpoint
pair from the coupling, place it on the path at a random time, and regress the
parametrization's target. What distinguishes score matching from flow matching
from bridge matching is which path, coupling and parametrization you hand it —
not which loss you call. :class:`EDMLoss` is a six-line configuration of
:class:`MatchingLoss` rather than an implementation, which is the check that the
factorization holds.

The coupling and the path meet here, and nowhere else. That join is the one
thing to get right: under the independent coupling x1 is fresh noise and the
score and noise targets are meaningful; under a bridge coupling x1 is a paired
endpoint, those targets are not, and the velocity target is what remains valid.

Pure PyTorch, tensor level. These take a model and a batch of samples, not a
SchNetPack batch dict; the adapter that maps atomistic batches onto this
contract, and the :class:`~schnetpack.objectives.UnsupervisedModelOutput` that
carries the result into the Lightning task, arrive with the atomistic port.
"""

from typing import Callable, Optional

import torch

from schnetpack.generative.couplings import Coupling, IndependentCoupling
from schnetpack.generative.parametrizations import Parametrization, X0Parametrization
from schnetpack.generative.paths import EDMPath, Path, expand_t

__all__ = ["MatchingLoss", "EDMLoss"]


class MatchingLoss:
    """
    Weighted regression of a parametrization's target along an interpolant path.

    Computes ``mean(w(t) (model(x_t, t) - target)^2)`` for x_t drawn from the
    path at times from ``t_sampler``. The path comes from the parametrization.
    """

    def __init__(
        self,
        parametrization: Parametrization,
        coupling: Optional[Coupling] = None,
        weight: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        t_sampler: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
    ):
        """
        Args:
            parametrization: what the model predicts, and hence what to regress;
                supplies the path
            coupling: how endpoint pairs are drawn (default: independent)
            weight: per-sample loss weight w(t), mapping (n_samples,) ->
                (n_samples,) (default: uniform)
            t_sampler: draws training times, mapping (n_samples, device) ->
                (n_samples,). The default is uniform on [t_min, t_max]. It
                stops short of t = 0 because the score target diverges there;
                the noise and denoiser targets are well behaved at 0, so if you
                train those on a VP path you may widen this back to [0, t_max].
        """
        self.parametrization = parametrization
        self.coupling = coupling if coupling is not None else IndependentCoupling()
        self.weight = weight if weight is not None else (lambda t: torch.ones_like(t))
        self.t_sampler = t_sampler if t_sampler is not None else self._uniform_t

    @property
    def path(self) -> Path:
        """The path being trained on, via the parametrization."""
        return self.parametrization.path

    def _uniform_t(self, n: int, device: Optional[torch.device]) -> torch.Tensor:
        span = self.path.t_max - self.path.t_min
        return self.path.t_min + span * torch.rand(n, device=device)

    def __call__(
        self,
        model: Callable,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        cond=None,
    ) -> torch.Tensor:
        """
        Args:
            model: callable (x, t, cond) -> raw output in ``parametrization``
            x0: clean samples, shape (n_samples, ...)
            x1: prior endpoints; drawn by the coupling if not given
            cond: conditioning passed through to the model

        Returns:
            Scalar loss.
        """
        x0, x1 = self.coupling.sample(x0, x1)
        t = self.t_sampler(x0.shape[0], x0.device)
        x_t = self.path.interpolate(x0, x1, t)

        prediction = model(x_t, t, cond)
        target = self.parametrization.target(x0, x1, t)
        w = expand_t(self.weight(t), x_t)
        return (w * (prediction - target) ** 2).mean()


class EDMLoss(MatchingLoss):
    """
    The EDM training objective (Karras et al. 2022).

    Two departures from the uniform default, both of which follow from the
    preconditioning rather than being free choices:

    - noise levels are drawn log-normally, concentrating training where the
      denoising problem is actually hard instead of spreading it evenly over a
      range spanning four orders of magnitude;
    - the weight is ``lambda(sigma) = (sigma^2 + sd^2) / (sigma sd)^2``, which
      is exactly ``1 / c_out(sigma)^2``. It cancels the preconditioner's output
      scaling so that every noise level contributes an effective loss of order
      one, and no level dominates the gradient.

    That second point is why ``sigma_data`` must match the one given to
    :class:`~schnetpack.generative.preconditioning.EDMPreconditioner`: the
    weighting and the preconditioner are two halves of one derivation. Nothing
    raises if they disagree — the model just trains against a badly scaled
    objective.

    Expects a model that predicts the clean sample, i.e. a
    :class:`~schnetpack.generative.preconditioning.PrecondDenoiser`.

    Note that the log-normal sampler is unbounded and ignores the path's
    ``t_min``/``t_max``, as in the paper: those bound *sampling*, while training
    benefits from the occasional extreme noise level.
    """

    def __init__(
        self,
        path: Optional[EDMPath] = None,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        """
        Args:
            path: sigma-space path (default: :class:`EDMPath`)
            P_mean: mean of log sigma
            P_std: standard deviation of log sigma
            sigma_data: data scale; must match the preconditioner's
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        super().__init__(
            parametrization=X0Parametrization(path if path is not None else EDMPath()),
            coupling=IndependentCoupling(),
            weight=self._edm_weight,
            t_sampler=self._lognormal_sigma,
        )

    def _lognormal_sigma(self, n: int, device: Optional[torch.device]) -> torch.Tensor:
        return torch.exp(self.P_mean + self.P_std * torch.randn(n, device=device))

    def _edm_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

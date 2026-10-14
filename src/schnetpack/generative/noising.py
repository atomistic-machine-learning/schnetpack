"""
Noising: how every noisy sample is paired with a clean structure.

Three levels, each a special case of the previous one:

- :class:`ForwardNoising`: any pairs (x0, x_noisy). Allows the denoiser
  and the pseudo force.
- :class:`Interpolant`: x_t = a(t) x0 + b(t) x1 with x1 drawn from a
  :class:`~schnetpack.generative.priors.Prior`, (x0, x1) paired by a
  :class:`~schnetpack.generative.couplings.Coupling` and (a, b) given by a
  :class:`~schnetpack.generative.schedules.Schedule`. Adds the velocity.
- :class:`GaussianInterpolant`: x_t = a(t) x0 + sigma(t) eps with a Gaussian
  prior and independent coupling. Adds the noise and the score.

Theory: ``docs_new/generation/noising.md``.
"""

import abc
import math

import torch

from schnetpack import properties
from schnetpack.generative.couplings import Coupling, IdentityCoupling
from schnetpack.generative.priors import GaussianPrior, Prior
from schnetpack.generative.schedules import GaussianSchedule, Schedule

__all__ = [
    "expand_t",
    "ForwardNoising",
    "Interpolant",
    "GaussianInterpolant",
]


def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Right-pad a per-sample time tensor so it broadcasts against sample data.

    Args:
        t: scalar (0-dim) or per-sample tensor of shape (n_samples,)
        x: data tensor of shape (n_samples, ...)
    """
    if t.dim() == 0:
        return t
    return t.reshape(t.shape[0], *([1] * (x.dim() - 1)))


class ForwardNoising(abc.ABC):
    """
    General noising: draws a noisy sample for each clean sample.

    No prior, schedule or time label is required, so only fields that are
    independent of them (the denoiser and the pseudo force) can be trained.
    """

    @abc.abstractmethod
    def perturb(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | None = None,
        batch=None,
        groups: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Draw a noisy sample for each clean sample.

        Args:
            x0: data batch, shape (n_samples, ...)
            t: noise times, per-sample or scalar, for noisings that use them
            batch: the batch x0 was taken from, for noisings that need the
                layout
            groups: per-row labels restricting which rows may be paired

        Returns:
            (x_noisy, x0, x1, t): the noisy batch, the (possibly re-paired)
            clean batch, the prior endpoint and the times; x1 and t are
            None for noisings that have none.
        """
        raise NotImplementedError


class Interpolant(ForwardNoising):
    """
    Interpolant x_t = a(t) x0 + b(t) x1.

    The schedule gives (a, b), their time derivatives and the time range
    [t_min, t_max]; the prior draws x1 and the coupling pairs (x0, x1).
    """

    def __init__(
        self,
        schedule: Schedule,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
    ):
        """
        Args:
            schedule: the schedule a(t), b(t)
            prior: distribution of the x1 endpoint (default: unit Gaussian)
            coupling: how endpoint batches are paired (default: independent)
        """
        self.schedule = schedule
        self.prior = prior if prior is not None else GaussianPrior(std=1.0)
        self.coupling = coupling if coupling is not None else IdentityCoupling()

    # -- noising ---------------------------------------------------------- #

    def interpolate(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Place a sample on the path: x_t = a x0 + b x1.

        Args:
            x0: data endpoint, shape (n_samples, ...)
            x1: prior endpoint, shaped like x0
            t: path time, per-sample or scalar
        """
        return (
            expand_t(self.schedule.a(t), x0) * x0
            + expand_t(self.schedule.b(t), x1) * x1
        )

    def perturb(self, x0, t=None, batch=None, groups=None):
        """
        Draw endpoints, pair them and place them on the path.

        Args:
            x0: data batch, shape (n_samples, ...)
            t: path times, per-sample or scalar; drawn from
                :meth:`sample_t` if not given
            batch: the batch x0 was taken from, handed to the prior (with
                x0 as its positions) so it can read the layout
            groups: per-row labels restricting which rows the coupling may
                exchange endpoints between; see
                :meth:`~schnetpack.generative.couplings.Coupling.pair`

        Returns:
            (x_t, x0, x1, t): the perturbed batch, the (possibly re-paired)
            endpoints and the times.
        """
        batch = {} if batch is None else batch
        x1 = self.prior({**batch, properties.R: x0})[properties.R]
        x0, x1 = self.coupling.pair(x0, x1, groups)
        if t is None:
            t = self.sample_t(x0.shape[0], x0.device).to(x0.dtype)
        return self.interpolate(x0, x1, t), x0, x1, t

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Default training-time density: uniform on [t_min, t_max].

        Other densities enter through the ``t_sampler`` hook of
        :class:`~schnetpack.generative.losses.MatchingLoss` and
        :class:`~schnetpack.generative.transforms.Diffuse`.
        """
        t_min, t_max = self.schedule.t_min, self.schedule.t_max
        return t_min + (t_max - t_min) * torch.rand(n, device=device)

    # -- sampling ----------------------------------------------------------#

    def sampling_prior(self) -> Prior:
        """
        Start distribution of the reverse process: the training prior itself.

        Raises:
            ValueError: if the coupling changes x1's marginal, in which case
                there is no data-free start and the sampler needs an
                explicit :class:`~schnetpack.generative.priors.Prior`.
        """
        if self.coupling.preserves_marginal:
            return self.prior
        raise ValueError(
            f"{type(self.coupling).__name__} reshapes x1's marginal from the "
            "data, so there is no data-free distribution to start sampling "
            "from. Pass an explicit Prior to the sampler — one matching the "
            "statistics this coupling trained under."
        )


class GaussianInterpolant(Interpolant):
    """
    Gaussian interpolant x_t = a(t) x0 + sigma(t) eps, eps ~ N(0, I).

    The prior is N(0, sigma_max^2 I) and the coupling is independent, so the
    conditional law p(x_t | x0) = N(a x0, sigma^2 I) is Gaussian. The
    schedule is a :class:`~schnetpack.generative.schedules.GaussianSchedule`,
    which defines sigma and derives b = sigma / sigma_max.
    """

    def __init__(self, schedule: GaussianSchedule, prior: Prior | None = None):
        """
        Args:
            schedule: the Gaussian schedule a(t), sigma(t)
            prior: Gaussian distribution of the x1 endpoint (default:
                N(0, sigma_max^2 I) with the schedule's sigma_max)

        Raises:
            TypeError: if the schedule is not a GaussianSchedule.
            ValueError: if the prior is not Gaussian, has no declared std,
                or its std differs from the schedule's sigma_max.
        """
        if not isinstance(schedule, GaussianSchedule):
            raise TypeError(
                f"GaussianInterpolant needs a GaussianSchedule, which defines "
                f"sigma(t); got {type(schedule).__name__}. Use an Interpolant "
                "for a plain schedule."
            )
        if prior is None:
            prior = GaussianPrior(std=schedule.sigma_max)
        if not prior.gaussian or prior.std is None:
            raise ValueError(
                f"{type(prior).__name__} is not an isotropic Gaussian with a "
                "declared std (prior.gaussian, prior.std), so x_t given x0 is "
                "not Gaussian. Use an Interpolant for a non-Gaussian prior."
            )
        if not math.isclose(prior.std, schedule.sigma_max):
            raise ValueError(
                f"The prior's std ({prior.std}) must equal the schedule's "
                f"sigma_max ({schedule.sigma_max})."
            )
        super().__init__(schedule, prior=prior, coupling=IdentityCoupling())

"""
Interpolant paths — the primitive every generative process is built from.

A path is the interpolant

    x_t = alpha(t) x0 + sigma(t) x1 + gamma(t) eps,

with x0 the data at t = 0 and x1 the prior endpoint at t = 1. The bridge noise
gamma is identically zero for every path here (see :meth:`Path.gamma`), so the
familiar two-term form x_t = alpha_t x0 + sigma_t x1 is what actually runs.

Subclasses implement four analytic schedules — alpha, sigma and their time
derivatives — and nothing else. Everything a path knows is derived from those:

- the forward SDE dx = f(t) x dt + g(t) dw, via f = alpha' / alpha and
  g^2 = 2 sigma sigma' - 2 f sigma^2;
- the closed-form marginals that make simulation-free training possible.

VE, VP, flow matching and EDM are therefore *configurations*, not classes with
bespoke logic.

This module is deliberately only geometry. Every method here is a function of
(alpha, sigma, alpha', sigma') and nothing else — a claim you can check by
reading one file. What the network *predicts*, the training targets and the
conversions between score, noise, denoiser and velocity all live on
:mod:`schnetpack.generative.parametrizations`, which binds to a path and reads
its schedule. Keeping the split means a new parametrization never touches this
file, and a new path never touches that one.

Why the interpolant is the primitive rather than (f, g): training needs x_t for
a random (x0, t) in one shot, which the interpolant gives and the SDE would
make you integrate. And (f, g) derive from it by inverting the mean and
variance ODEs, whereas the converse costs an ODE solve per path — and for flow
matching there is no intrinsic g to start from, since its diffusion is a
sampler choice.

Times may be per-sample (shape ``(n_samples,)``): samples in one batch can sit
at different path times, which per-sample training and adaptive integrators
rely on.
"""

import abc
import math
from typing import Optional, Tuple

import torch

__all__ = [
    "expand_t",
    "Path",
    "VPPath",
    "VEPath",
    "VELinearPath",
    "FMPath",
    "EDMPath",
]


def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Right-pad a per-sample time tensor for broadcasting against sample data.

    Args:
        t: scalar (0-dim) or per-sample tensor of shape (n_samples,)
        x: data tensor of shape (n_samples, ...)
    """
    if t.dim() == 0:
        return t
    return t.reshape(t.shape[0], *([1] * (x.dim() - 1)))


class Path(abc.ABC):
    """
    Interpolant x_t = alpha(t) x0 + sigma(t) x1 + gamma(t) eps on [t_min, t_max].

    Subclasses implement :meth:`alpha`, :meth:`sigma` and their derivatives;
    drift, diffusion, marginals, targets and conversions all derive from those.

    ``t_min`` and ``t_max`` bound the usable time range and are consumed by
    prior sampling, the default training time sampler and the sampler grids.
    They exist because the endpoints are where paths misbehave: the score
    diverges as sigma -> 0, and some diffusion coefficients blow up at t_max.
    """

    def __init__(self, t_min: float, t_max: float):
        """
        Args:
            t_min: smallest usable path time
            t_max: largest usable path time (where the prior is drawn)
        """
        self.t_min = t_min
        self.t_max = t_max

    # -- schedules -------------------------------------------------------- #

    @abc.abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Data coefficient alpha(t), shaped like t."""
        raise NotImplementedError

    @abc.abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient sigma(t), shaped like t."""
        raise NotImplementedError

    @abc.abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`alpha`, shaped like t."""
        raise NotImplementedError

    @abc.abstractmethod
    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`sigma`, shaped like t."""
        raise NotImplementedError

    def gamma(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Bridge noise coefficient, or None when it vanishes identically.

        None rather than a zero tensor: it lets :meth:`interpolate` skip the
        term entirely instead of drawing noise and multiplying it by zero.
        Bridge paths (Schrödinger bridge, and any interpolant whose x_t depends
        on both endpoints plus its own noise) override this; nothing else needs
        to.
        """
        return None

    # -- derived scalars -------------------------------------------------- #

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient f(t) = alpha'(t) / alpha(t) of the forward SDE."""
        return self.alpha_dot(t) / self.alpha(t)

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion g^2(t) = 2 sigma sigma' - 2 f sigma^2.

        This is the diffusion of the forward SDE that shares the path's
        marginals. It is a *canonical* choice rather than an intrinsic property
        of the interpolant: any g gives the same marginals as long as the drift
        matches, and the sampler picks how much of it to use via its churn knob.
        Flow matching runs at churn = 0 and never touches this.
        """
        sigma = self.sigma(t)
        return 2.0 * sigma * self.sigma_dot(t) - 2.0 * self.f(t) * sigma**2

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both marginal coefficients at once, each shaped like t."""
        return self.alpha(t), self.sigma(t)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio alpha^2 / sigma^2."""
        return self.alpha(t) ** 2 / self.sigma(t) ** 2

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Log signal-to-noise ratio, 2 (log alpha - log sigma)."""
        return 2.0 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    # -- interpolant and marginals ---------------------------------------- #

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Place a sample on the path: x_t = alpha x0 + sigma x1 (+ gamma eps).

        Args:
            x0: data endpoint, shape (n_samples, ...)
            x1: prior endpoint, shape of x0; Gaussian noise for the
                independent coupling, a paired endpoint for bridges
            t: path time, per-sample or scalar
            eps: bridge noise; drawn if needed and not given, ignored when
                :meth:`gamma` returns None
        """
        x_t = expand_t(self.alpha(t), x0) * x0 + expand_t(self.sigma(t), x1) * x1
        gamma = self.gamma(t)
        if gamma is None:
            return x_t
        if eps is None:
            eps = torch.randn_like(x0)
        return x_t + expand_t(gamma, eps) * eps

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and std of the transition kernel p(x_t | x0) = N(alpha x0, sigma^2 I).

        Only meaningful for Gaussian x1 (the independent coupling).

        Returns:
            mean shaped like x0, std shaped like t.
        """
        return expand_t(self.alpha(t), x0) * x0, self.sigma(t)

    def diffuse(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ p(x_t | x0) in one shot — the simulation-free noising step.

        Returns:
            (x_t, noise), both shaped like x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return self.interpolate(x0, noise, t), noise


class VPPath(Path):
    """
    Variance-preserving path (continuous-time DDPM) with linear beta.

    alpha = exp(-1/2 int_0^t beta), sigma = sqrt(1 - alpha^2), which gives
    f = -1/2 beta and g^2 = beta. Unit-variance data keeps unit variance for
    all t, and p_{t_max} -> N(0, I).
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ):
        """
        Args:
            beta_min: beta(0)
            beta_max: beta(t_max)
            t_min: smallest usable time; the score diverges as t -> 0
            t_max: largest usable time
        """
        super().__init__(t_min=t_min, t_max=t_max)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (t / self.t_max) * (self.beta_max - self.beta_min)

    def _log_alpha(self, t: torch.Tensor) -> torch.Tensor:
        int_beta = (
            self.beta_min * t
            + 0.5 * (self.beta_max - self.beta_min) * t**2 / self.t_max
        )
        return -0.5 * int_beta

    def alpha(self, t):
        return torch.exp(self._log_alpha(t))

    def alpha_dot(self, t):
        return -0.5 * self.beta(t) * self.alpha(t)

    def sigma(self, t):
        # sqrt(-expm1(2 log alpha)) rather than sqrt(1 - alpha^2): at small t
        # alpha^2 is within an ulp of 1 and the naive form loses every digit.
        return torch.sqrt(-torch.expm1(2.0 * self._log_alpha(t)))

    def sigma_dot(self, t):
        # d/dt sqrt(1 - alpha^2) = -alpha alpha' / sigma
        alpha = self.alpha(t)
        return -alpha * self.alpha_dot(t) / torch.clamp(self.sigma(t), min=1e-12)


class VEPath(Path):
    """
    Variance-exploding path with geometric sigma (score matching / SMLD).

    alpha = 1 and sigma(t) = sigma_min (sigma_max/sigma_min)^(t/t_max), so the
    mean never moves and p_{t_max} ~= N(0, sigma_max^2 I). Since sigma(0) =
    sigma_min > 0, nothing is singular at t = 0 and t_min may stay there.

    Unlike VP, this path is *not* scale-free: sigma_max has to match your data.
    The rule of thumb (Song & Ermon 2020) is the largest pairwise distance in
    the dataset — enough to drown out the data, and no more. Overshooting does
    not fail loudly; it spends most of the time range on noise levels where
    there is nothing left to learn, and a model trained there samples badly
    while its loss looks fine. If sampling produces garbage, check this first.
    The default suits data of order 10 (atomic positions in Angstrom); for
    unit-scale data try sigma_max ~ 3.

    Predicting the score directly here also wants ``weight=lambda t:
    path.sigma(t)**2`` in the loss: sigma spans orders of magnitude, so the
    -x1/sigma target does too, and an unweighted L2 sees only its low-noise
    end. That weighting makes the objective identical to eps-matching, which is
    the other way to get the same result.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        """
        Args:
            sigma_min: noise level at t = 0
            sigma_max: noise level at t = t_max; must be comparable to the
                spread of the data (see the class docstring)
            t_min: smallest usable time
            t_max: largest usable time
        """
        super().__init__(t_min=t_min, t_max=t_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def alpha(self, t):
        return torch.ones_like(t)

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.t_max)

    def sigma_dot(self, t):
        return self.sigma(t) * math.log(self.sigma_max / self.sigma_min) / self.t_max


class VELinearPath(Path):
    """
    Variance-exploding path with linear sigma: alpha = 1, sigma(t) = t sigma_max.

    This is :class:`EDMPath` mapped onto [0, 1] — same geometry, different time
    coordinate. Prefer EDMPath when you want Karras-style sigma-space sampling
    and preconditioning; this one keeps the [0, 1] convention shared with the
    other paths.
    """

    def __init__(
        self, sigma_max: float = 80.0, t_min: float = 1e-3, t_max: float = 1.0
    ):
        """
        Args:
            sigma_max: noise level at t = t_max
            t_min: smallest usable time; sigma -> 0 as t -> 0
            t_max: largest usable time
        """
        super().__init__(t_min=t_min, t_max=t_max)
        self.sigma_max = sigma_max

    def alpha(self, t):
        return torch.ones_like(t)

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def sigma(self, t):
        return t * self.sigma_max

    def sigma_dot(self, t):
        return torch.full_like(t, self.sigma_max)


class FMPath(Path):
    """
    Linear interpolant for flow matching / rectified flow:
    alpha = 1 - t, sigma = t sigma_max.

    The velocity target is constant along a pair, x1 - x0 up to scale, which is
    what makes the learned field straight and few-step sampling work.

    The prior at t = 1 is exact here (alpha(1) = 0), but g^2 = 2 t sigma_max^2 /
    (1 - t) diverges there, so ``t_max`` defaults just below 1. That costs a
    dropped alpha(t_max) x0 term of order 1e-3 times the data scale in
    :meth:`sample_prior` — negligible, but it is why the default is not exactly
    1. Pure-ODE users (churn = 0, where g^2 is never evaluated) can pass
    ``t_max=1.0`` and get the exact prior.
    """

    def __init__(
        self,
        sigma_max: float = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
    ):
        """
        Args:
            sigma_max: noise scale at t = 1
            t_min: smallest usable time; sigma -> 0 as t -> 0
            t_max: largest usable time; keep < 1 for stochastic sampling
        """
        super().__init__(t_min=t_min, t_max=t_max)
        self.sigma_max = sigma_max

    def alpha(self, t):
        return 1.0 - t

    def alpha_dot(self, t):
        return -torch.ones_like(t)

    def sigma(self, t):
        return t * self.sigma_max

    def sigma_dot(self, t):
        return torch.full_like(t, self.sigma_max)


class EDMPath(Path):
    """
    Variance-exploding path in sigma-coordinates: alpha = 1, sigma(t) = t.

    Time *is* the noise level here — the [0, 1] convention of the other paths is
    deliberately dropped, following Karras et al. (2022), because decoupling the
    noise level from a nominal time is exactly what lets the schedule, the
    preconditioning and the sampler grid be chosen independently. f = 0 and
    g^2 = 2 sigma.

    Pairs with :class:`~schnetpack.generative.preconditioning.EDMPreconditioner`
    and :class:`~schnetpack.generative.grids.KarrasGrid`.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        """
        Args:
            sigma_min: smallest noise level (= t_min)
            sigma_max: largest noise level (= t_max), where the prior is drawn
        """
        super().__init__(t_min=sigma_min, t_max=sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def alpha(self, t):
        return torch.ones_like(t)

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def sigma(self, t):
        return t

    def sigma_dot(self, t):
        return torch.ones_like(t)

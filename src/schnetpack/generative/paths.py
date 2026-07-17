"""
Interpolant paths — the primitive every generative process is built from.

A path is the interpolant

    x_t = alpha(t) x0 + sigma(t) x1 + gamma(t) eps,

with x0 the data at t = 0 and x1 the prior endpoint at t = 1. The bridge noise
gamma is identically zero for every path here (see :meth:`Path.gamma`), so the
familiar two-term form x_t = alpha_t x0 + sigma_t x1 is what actually runs.

A subclass fixes the schedule by defining **one** of two pairs:

- ``alpha`` and ``sigma``, the coefficients directly; or
- ``tv`` and ``log_snr`` — the total variance alpha^2 + sigma^2 and the log
  signal-to-noise ratio log(alpha^2 / sigma^2).

The pairs are equivalent, and each derives the other, so a path always answers
all four. The second is the TV/SNR reparametrization of Kahouli et al. (2025),
arXiv:2502.08598, and it exists because the two quantities are *independent*
knobs where alpha and sigma are not: TV says how large x_t is, SNR says how much
of it is signal, and moving one leaves the other alone. Inverting is a two-liner
(see :meth:`Path.alpha`), which is what lets both routes sit in one base class.

Time derivatives are not asked for. ``alpha_dot`` and ``sigma_dot`` default to
autograd through whichever schedule was written, so a new path is two functions
and nothing else. The paths here override them analytically because they can —
it is faster and exact — and a test pins that the two agree.

Everything else a path knows is derived from those:

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
import inspect
import math
from typing import Callable, Optional, Tuple

import torch

__all__ = [
    "expand_t",
    "Path",
    "VPPath",
    "VEPath",
    "VELinearPath",
    "FMPath",
    "EDMPath",
    "VPISSNRPath",
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

    A subclass fixes the schedule by defining **either** of two pairs:

    - :meth:`alpha` and :meth:`sigma` — the coefficients directly;
    - :meth:`tv` and :meth:`log_snr` — the total variance alpha^2 + sigma^2 and
      the log signal-to-noise ratio log(alpha^2 / sigma^2).

    Whichever pair is given, the other is derived (see :meth:`alpha`), so both
    are always available. Defining neither is a TypeError at class-definition
    time — the two routes are mutually recursive, and this is where that shows
    up as a readable error rather than a RecursionError at the first call.

    The TV/SNR route is the reparametrization of Kahouli et al. (2025),
    "Total-Variance/Signal-to-Noise-Ratio Disentangled Diffusion"
    (arXiv:2502.08598). The point is that the two knobs are independent: TV
    fixes how large x_t is, SNR fixes how much of it is signal, and neither
    constrains the other. Written as alpha/sigma those choices are tangled —
    changing the noise level moves the total variance too — which is why the
    schedules that work are folklore. See :class:`VPISSNRPath`.

    The derivatives :meth:`alpha_dot` and :meth:`sigma_dot` default to autograd
    through the schedule, so a subclass need not supply them. Overriding them
    with analytic forms is a speed and precision optimization, not a
    requirement; every path in this module does, and their agreement with
    autograd is tested.

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
        if type(self) is Path:
            raise TypeError(
                "Path is abstract. Subclass it and define either (alpha, sigma) "
                "or (tv, log_snr)."
            )
        self.t_min = t_min
        self.t_max = t_max

    def __init_subclass__(cls, **kwargs):
        """
        Reject a subclass that defines neither schedule pair.

        :meth:`alpha`/:meth:`sigma` and :meth:`tv`/:meth:`log_snr` are defined
        in terms of each other, so a subclass supplying neither would recurse
        until the stack ran out — on first use, far from the cause. Catching it
        here turns that into a TypeError naming the class that is wrong.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        has_alpha_sigma = cls.alpha is not Path.alpha and cls.sigma is not Path.sigma
        has_tv_snr = cls.tv is not Path.tv and cls.log_snr is not Path.log_snr
        if not (has_alpha_sigma or has_tv_snr):
            raise TypeError(
                f"{cls.__name__} defines no schedule. A Path must implement "
                "either (alpha, sigma) or (tv, log_snr); each pair derives the "
                "other. Note both members of a pair are needed: defining only "
                "one leaves the derivation of the other pair recursive."
            )

    # -- schedules -------------------------------------------------------- #

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Data coefficient alpha(t), shaped like t.

        Derived from the TV/SNR pair unless overridden. Inverting
        TV = alpha^2 + sigma^2 and SNR = alpha^2 / sigma^2 gives

            alpha^2 = TV * SNR / (1 + SNR) = TV * sigmoid(log SNR),

        using SNR / (1 + SNR) = sigmoid(log SNR). Going through the sigmoid is
        not cosmetic: SNR itself spans the whole positive axis and overflows a
        float at the ends of a schedule, where sigmoid(log SNR) simply
        saturates at 1.

        Assumes alpha >= 0, which the square root cannot recover on its own.
        """
        return torch.sqrt(self.tv(t) * torch.sigmoid(self.log_snr(t)))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Noise coefficient sigma(t), shaped like t.

        The mirror of :meth:`alpha`: sigma^2 = TV / (1 + SNR) =
        TV * sigmoid(-log SNR).
        """
        return torch.sqrt(self.tv(t) * torch.sigmoid(-self.log_snr(t)))

    def tv(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total variance alpha^2 + sigma^2, shaped like t.

        The variance of x_t for unit-variance data and noise: how big the
        interpolated sample is, independent of how much of it is signal. A
        variance-preserving path is exactly one with tv == 1.

        Derived from alpha/sigma unless overridden.
        """
        return self.alpha(t) ** 2 + self.sigma(t) ** 2

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Log signal-to-noise ratio, 2 (log alpha - log sigma).

        The other half of the TV/SNR pair, and the one to define: it is finite
        and well conditioned over the whole schedule where :meth:`snr` itself
        overflows. If you have the ratio rather than its log, return
        ``torch.log(gamma(t))``.

        Derived from alpha/sigma unless overridden. Note the convention:
        SNR = alpha^2 / sigma^2 is the *squared* ratio, matching the gamma of
        the TV/SNR reference implementation (whose ``log_gamma`` is this
        function) and Kingma's log-SNR.
        """
        return 2.0 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio alpha^2 / sigma^2."""
        return self.alpha(t) ** 2 / self.sigma(t) ** 2

    # -- derivatives ------------------------------------------------------ #

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`alpha`, shaped like t (autograd default)."""
        return self._time_derivative(self.alpha, t)

    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`sigma`, shaped like t (autograd default)."""
        return self._time_derivative(self.sigma, t)

    def log_alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log alpha(t), shaped like t. Equals alpha_dot / alpha.

        This is the hook the identity buys, and :meth:`f` is exactly it. The
        default here is honest about its limits: autograd through ``log(alpha)``
        applies the chain rule as (1 / alpha) * alpha_dot, so it *is* the
        quotient and degenerates in all the same places. What it gives you is a
        place to override.

        Override it whenever log alpha is known in closed form, because that
        form is usually the one with no division in it — VP's is simply
        -beta/2, finite everywhere including where alpha underflows to zero and
        the quotient becomes 0/0. Every path in this module does so.
        """
        return self._log_derivative(self.alpha, t)

    def log_sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt log sigma(t). Equals sigma_dot / sigma; see :meth:`log_alpha_dot`."""
        return self._log_derivative(self.sigma, t)

    def log_snr_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log SNR(t), shaped like t.

        Differentiates :meth:`log_snr` directly rather than differencing
        :meth:`log_alpha_dot` and :meth:`log_sigma_dot`: one pass instead of
        two, and — for a path defined the TV/SNR way — the derivative of the
        closed form the subclass actually wrote, with no alpha or sigma formed
        along the way and so no quotient to degenerate. That is what makes
        :meth:`g2` well behaved on schedules whose alpha reaches zero.

        Non-positive for any sensible path — signal only ever turns into noise —
        which is what makes :meth:`g2` non-negative.
        """
        return self._time_derivative(self.log_snr, t)

    @staticmethod
    def _log_derivative(
        fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """d/dt log fn(t), by autograd through the log rather than by dividing."""
        return Path._time_derivative(lambda s: torch.log(fn(s)), t)

    @staticmethod
    def _time_derivative(
        fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """
        d fn / dt by autograd, so a schedule need not be differentiated by hand.

        Differentiates ``fn(t).sum()``, which gives the per-sample derivative
        only because schedules are elementwise in t — sample i's value does not
        depend on sample j's time, so the sum's gradient is the vector of
        individual derivatives. A schedule coupling times across the batch would
        silently get the wrong answer here and must override.

        ``enable_grad`` because this runs inside ``torch.no_grad()`` during
        sampling, where the graph would otherwise never be built.

        The result is a plain tensor: ``t`` is detached going in and
        ``create_graph`` is False, so nothing here carries a graph back out.
        That matters beyond tidiness — :class:`~schnetpack.generative.transforms.Diffuse`
        builds targets inside the dataloader, and a target holding a grad_fn
        cannot be pickled to a worker process. The cost is that a schedule with
        *learnable* parameters gets no gradient through this; such a path should
        override the derivative analytically.

        A constant schedule (``alpha = ones_like(t)``) produces a tensor with no
        grad_fn at all, which autograd reports as unused rather than as zero —
        hence the two zero fallbacks.
        """
        with torch.enable_grad():
            t_ = t.detach().clone().requires_grad_(True)
            y = fn(t_)
            if not y.requires_grad:
                return torch.zeros_like(t)
            (grad,) = torch.autograd.grad(y.sum(), t_, allow_unused=True)
        return torch.zeros_like(t) if grad is None else grad

    def gamma(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Bridge noise coefficient, or None when it vanishes identically.

        None rather than a zero tensor: it lets :meth:`interpolate` skip the
        term entirely instead of drawing noise and multiplying it by zero.
        Bridge paths (Schrödinger bridge, and any interpolant whose x_t depends
        on both endpoints plus its own noise) override this; nothing else needs
        to.

        Not to be confused with the gamma of the TV/SNR literature, which is the
        signal-to-noise ratio and lives here as :meth:`snr` / :meth:`log_snr`.
        This name predates that and means the third interpolant coefficient.
        """
        return None

    # -- derived scalars -------------------------------------------------- #

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient f(t) = alpha'/alpha of the forward SDE.

        Which is d/dt log alpha — see :meth:`log_alpha_dot`, where the quotient
        is avoided rather than computed.
        """
        return self.log_alpha_dot(t)

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion of the forward SDE, g^2 = -sigma^2 d/dt log SNR.

        The usual form is g^2 = 2 sigma sigma' - 2 f sigma^2. Factoring sigma^2
        out of both terms leaves the two log-derivatives,

            g^2 = 2 sigma^2 (d/dt log sigma - d/dt log alpha),

        and log SNR = 2 (log alpha - log sigma) makes that bracket exactly
        -1/2 d/dt log SNR. So the whole diffusion is one log-derivative of one
        schedule — no quotient, and nothing that degenerates where alpha or
        sigma vanish. It also puts the sign where it can be read: g^2 >= 0
        precisely because SNR decreases.

        This is the diffusion that shares the path's marginals. It is a
        *canonical* choice rather than an intrinsic property of the
        interpolant: any g gives the same marginals as long as the drift
        matches, and the sampler picks how much of it to use via its churn knob.
        Flow matching runs at churn = 0 and never touches this.
        """
        return -self.sigma(t) ** 2 * self.log_snr_dot(t)

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both marginal coefficients at once, each shaped like t."""
        return self.alpha(t), self.sigma(t)

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

    def log_alpha_dot(self, t):
        # d/dt (-1/2 int beta) -- the textbook f, exact and division-free
        return -0.5 * self.beta(t)

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

    def log_alpha_dot(self, t):
        return torch.zeros_like(t)  # alpha == 1

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

    def log_alpha_dot(self, t):
        return torch.zeros_like(t)  # alpha == 1

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

    def log_alpha_dot(self, t):
        return -1.0 / (1.0 - t)

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

    def log_alpha_dot(self, t):
        return torch.zeros_like(t)  # alpha == 1

    def sigma(self, t):
        return t

    def sigma_dot(self, t):
        return torch.ones_like(t)


class VPISSNRPath(Path):
    """
    Variance-preserving path with an inverse-sigmoid log-SNR (arXiv:2502.08598).

    The headline schedule of the TV/SNR paper, and the one path here defined the
    TV/SNR way rather than the alpha/sigma way::

        TV(t)      = 1                          (variance preserving)
        log SNR(t) = eta * log(1/t - 1) + kappa

    alpha and sigma follow from :meth:`Path.alpha`, and the derivatives from
    autograd — nothing else is written. That is the point of the class: it is
    the whole schedule, and it exists to show that the second route is a real
    one rather than a convenience.

    Writing it this way is what makes the two knobs mean something separately.
    ``eta`` sets how fast signal turns into noise and ``kappa`` shifts where the
    schedule sits, while TV = 1 pins the total variance no matter what either
    does. In alpha/sigma coordinates that separation does not exist: any change
    to the noise level moves the total variance too, and you have to fix it back
    up by hand.

    log(1/t - 1) is the inverse of the sigmoid, so alpha^2 = sigmoid(log SNR)
    runs smoothly from ~1 to ~0 with no endpoint chosen by fiat — but it also
    diverges at t = 0 and t = 1, which is why ``t_min`` and ``t_max`` sit
    strictly inside.

    ``eta = 2`` is the natural default: with the paper's companion TV schedule
    ``(1-t)^eta + t^eta exp(-kappa)`` the same inversion gives alpha^2 =
    (1-t)^eta and sigma^2 = t^eta exp(-kappa), so eta = 2, kappa = 0 is exactly
    optimal-transport flow matching (:class:`FMPath`). This class is its
    variance-preserving sibling: same SNR schedule, TV flattened to 1.

    (The paper quotes eta = 1 for that correspondence because its text defines
    SNR as alpha/sigma, while its code — and :meth:`Path.log_snr` — uses the
    squared alpha^2/sigma^2. The factor of two lands on eta.)
    """

    def __init__(
        self,
        eta: float = 2.0,
        kappa: float = 0.0,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
    ):
        """
        Args:
            eta: steepness of the log-SNR; > 0. eta = 2 matches flow matching.
            kappa: shift of the log-SNR; > 0 moves the schedule toward signal,
                < 0 toward noise
            t_min: smallest usable time; the log-SNR diverges at t = 0
            t_max: largest usable time; the log-SNR diverges at t = 1
        """
        super().__init__(t_min=t_min, t_max=t_max)
        self.eta = eta
        self.kappa = kappa

    def tv(self, t):
        return torch.ones_like(t)

    def log_snr(self, t):
        return self.eta * torch.log(1.0 / t - 1.0) + self.kappa

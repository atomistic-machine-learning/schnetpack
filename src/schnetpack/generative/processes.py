"""
Forward processes: schedule, prior and coupling in one object.

A process is the interpolant x_t = a(t) x0 + b(t) x1 with x1 drawn from a
:class:`~schnetpack.generative.priors.Prior` and (x0, x1) paired by a
:class:`~schnetpack.generative.couplings.Coupling`. Subclasses define only
the schedule, either as (a, b) or as (tv, log_snr); prior and coupling are
constructor arguments. Theory and design: ``docs_new/processes.md``.
"""

import abc
import inspect
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from schnetpack import properties

if TYPE_CHECKING:
    from schnetpack.generative.differential_equations import SDE

from schnetpack.generative.couplings import Coupling, IdentityCoupling
from schnetpack.generative.priors import GaussianPrior, Prior

__all__ = [
    "expand_t",
    "Process",
    "VP",
    "VE",
    "VELinear",
    "FlowMatching",
    "VPISSNR",
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


class Process(abc.ABC):  # noqa: B024 - schedule enforced in __init_subclass__
    """
    Forward process x_t = a(t) x0 + b(t) x1 (+ gamma(t) eps) on [t_min, t_max].

    b is a dimensionless blending weight in [0, 1] with b(t_max) = 1; the
    endpoint's scale lives on the prior, and the noise level is
    sigma(t) = b(t) * prior.std. A subclass defines either (:meth:`a`,
    :meth:`b`) or (:meth:`tv`, :meth:`log_snr`); the other pair is derived.
    Time derivatives default to autograd and may be overridden analytically.
    """

    def __init__(
        self,
        t_min: float,
        t_max: float,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
        scale: float | None = None,
    ):
        """
        Args:
            t_min: smallest usable path time
            t_max: largest usable path time, where the prior is drawn
            prior: distribution of the x1 endpoint (default: unit Gaussian);
                mutually exclusive with ``scale``
            coupling: how endpoint batches are paired (default: identity)
            scale: shorthand for ``prior=GaussianPrior(std=scale)``
        """
        if type(self) is Process:
            raise TypeError(
                "Process is abstract. Subclass it and define either (a, b) "
                "or (tv, log_snr)."
            )
        if prior is not None and scale is not None:
            raise TypeError(
                "Pass either scale or prior, not both — scale is shorthand "
                "for prior=GaussianPrior(std=scale)."
            )
        self.t_min = t_min
        self.t_max = t_max
        self.prior = (
            prior
            if prior is not None
            else GaussianPrior(std=1.0 if scale is None else scale)
        )
        self.coupling = coupling if coupling is not None else IdentityCoupling()

    def __init_subclass__(cls, **kwargs):
        """Reject a concrete subclass that defines neither schedule pair."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        has_a_b = cls.a is not Process.a and cls.b is not Process.b
        has_tv_snr = cls.tv is not Process.tv and cls.log_snr is not Process.log_snr
        if not (has_a_b or has_tv_snr):
            raise TypeError(
                f"{cls.__name__} defines no schedule. A Process must "
                "implement either (a, b) or (tv, log_snr); each pair derives "
                "the other. Note both members of a pair are needed: defining "
                "only one leaves the derivation of the other pair recursive."
            )

    # -- schedules -------------------------------------------------------- #

    def a(self, t: torch.Tensor) -> torch.Tensor:
        """Data coefficient a(t), shaped like t. Derived from (tv, log_snr) unless overridden."""
        # a^2 = TV * SNR / (1 + SNR) = TV * sigmoid(log SNR); the sigmoid form
        # does not overflow where SNR itself would.
        return torch.sqrt(self.tv(t) * torch.sigmoid(self.log_snr(t)))

    def b(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient b(t), shaped like t. Derived from (tv, log_snr) unless overridden."""
        return torch.sqrt(self.tv(t) * torch.sigmoid(-self.log_snr(t)))

    def tv(self, t: torch.Tensor) -> torch.Tensor:
        """Total variance a^2 + b^2, shaped like t. Derived from (a, b) unless overridden."""
        return self.a(t) ** 2 + self.b(t) ** 2

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Log signal-to-noise ratio log(a^2 / b^2). Derived from (a, b) unless overridden."""
        return 2.0 * (torch.log(self.a(t)) - torch.log(self.b(t)))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio a^2 / b^2."""
        return self.a(t) ** 2 / self.b(t) ** 2

    # -- derivatives ------------------------------------------------------ #

    def a_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`a` (autograd unless overridden)."""
        return self._time_derivative(self.a, t)

    def b_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`b` (autograd unless overridden)."""
        return self._time_derivative(self.b, t)

    def log_a_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log a(t) = a_dot / a, the drift f of the SDE chart.

        Override with the closed form where one exists: it is usually
        division-free and stays finite where a underflows to zero.
        """
        return self._log_derivative(self.a, t)

    def log_b_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt log b(t) = b_dot / b; see :meth:`log_a_dot`."""
        return self._log_derivative(self.b, t)

    def log_snr_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log SNR(t), shaped like t; non-positive for any sensible schedule.

        Differentiated directly rather than as log_a_dot - log_b_dot, so a
        schedule defined via (tv, log_snr) never forms a or b on the way.
        """
        return self._time_derivative(self.log_snr, t)

    @staticmethod
    def _log_derivative(
        fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """d/dt log fn(t) by autograd through the log."""
        return Process._time_derivative(lambda s: torch.log(fn(s)), t)

    @staticmethod
    def _time_derivative(
        fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """
        d fn / dt by autograd, returned as a plain tensor without a graph.

        Assumes ``fn`` is elementwise in t (the gradient of ``fn(t).sum()``
        is then the per-sample derivative). Works under ``torch.no_grad()``.
        A schedule with learnable parameters gets no gradient through this
        and should override the derivative analytically.
        """
        with torch.enable_grad():
            t_ = t.detach().clone().requires_grad_(True)
            y = fn(t_)
            if not y.requires_grad:
                # constant schedule: no graph at all
                return torch.zeros_like(t)
            (grad,) = torch.autograd.grad(y.sum(), t_, allow_unused=True)
        return torch.zeros_like(t) if grad is None else grad

    def gamma(self, t: torch.Tensor) -> torch.Tensor | None:
        """
        Bridge noise coefficient, or None when it vanishes identically.

        None lets :meth:`interpolate` skip the term. Bridge processes
        override this; nothing else needs to. (Not the gamma of the TV/SNR
        literature, which is :meth:`snr` here.)
        """
        return None

    def a_b(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both marginal coefficients at once, each shaped like t."""
        return self.a(t), self.b(t)

    # -- scale: the prior's declaration, exposed once ---------------------- #

    @property
    def std(self) -> float:
        """
        Endpoint scale ``prior.std``.

        Raises:
            ValueError: if the prior declares no scalar scale.
        """
        if self.prior.std is None:
            raise ValueError(
                f"{type(self.prior).__name__} declares no scalar endpoint "
                "scale (std is None), so this process has no single noise "
                "level sigma(t) = b(t) * std. Score conversions, the SDE "
                "diffusion and churn > 0 sampling all need one — use a prior "
                "with a declared std, or stick to the routes that never form "
                "sigma (the pseudo-force x0 recovery, churn = 0 velocity "
                "sampling)."
            )
        return self.prior.std

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise level sigma(t) = b(t) * prior.std."""
        return self.b(t) * self.std

    def t_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Inverse of :meth:`sigma`: the path time at which a noise level sits.

        Bisection over [t_min, t_max], which only needs b to be monotone;
        subclasses with a closed form override it.

        Args:
            sigma: noise levels, any shape

        Returns:
            Times of the same shape, clamped to [t_min, t_max].
        """
        sigma = torch.as_tensor(sigma)
        lo = torch.full_like(sigma, self.t_min, dtype=torch.float64)
        hi = torch.full_like(sigma, self.t_max, dtype=torch.float64)
        target = sigma.to(torch.float64)
        # b is monotone but its direction is the schedule's business: read it
        # off the endpoints rather than assuming noise grows with t.
        ends = torch.tensor([self.t_min, self.t_max], dtype=torch.float64)
        s_min, s_max = self.sigma(ends).to(torch.float64)
        increasing = bool(s_max >= s_min)
        for _ in range(60):  # float64 exhausted; 60 halvings of [0, 1]
            mid = 0.5 * (lo + hi)
            below = self.sigma(mid.to(sigma.dtype)).to(torch.float64) < target
            take_upper = below if increasing else ~below
            lo = torch.where(take_upper, mid, lo)
            hi = torch.where(take_upper, hi, mid)
        return (0.5 * (lo + hi)).to(sigma.dtype).clamp(self.t_min, self.t_max)

    # -- interpolant ------------------------------------------------------ #

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Place a sample on the path: x_t = a x0 + b x1 (+ gamma eps).

        Args:
            x0: data endpoint, shape (n_samples, ...)
            x1: prior endpoint, shaped like x0
            t: path time, per-sample or scalar
            eps: bridge noise; drawn if needed and not given, ignored when
                :meth:`gamma` returns None
        """
        x_t = expand_t(self.a(t), x0) * x0 + expand_t(self.b(t), x1) * x1
        gamma = self.gamma(t)
        if gamma is None:
            return x_t
        if eps is None:
            eps = torch.randn_like(x0)
        return x_t + expand_t(gamma, eps) * eps

    # -- the forward move --------------------------------------------------#

    def perturb(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        context=None,
        groups: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Draw endpoints, pair them and place them on the path.

        Owns every random draw of the forward side, so the training target
        can be built from exactly the draws that made x_t.

        Args:
            x0: data batch, shape (n_samples, ...)
            x1: endpoint batch to use instead of drawing from the prior;
                still passed through the coupling
            t: path times, per-sample or scalar; drawn from
                :meth:`sample_t` if not given
            context: the batch x0 was taken from, handed to the prior (with
                x0 as its positions) so it can read the layout
            groups: per-row labels restricting which rows the coupling may
                exchange endpoints between; see
                :meth:`~schnetpack.generative.couplings.Coupling.pair`

        Returns:
            (x_t, x0, x1, t, eps): the perturbed batch, the (possibly
            re-paired) endpoints, the times and the bridge noise (None when
            gamma is zero).
        """
        if x1 is None:
            batch = {} if context is None else context
            x1 = self.prior.sample_positions({**batch, properties.R: x0})
        # only passed when asked for, so couplings written against the
        # two-argument `pair` keep working
        if groups is None:
            x0, x1 = self.coupling.pair(x0, x1)
        else:
            x0, x1 = self.coupling.pair(x0, x1, groups)
        if t is None:
            t = self.sample_t(x0.shape[0], x0.device).to(x0.dtype)
        eps = None if self.gamma(t) is None else torch.randn_like(x0)
        x_t = self.interpolate(x0, x1, t, eps=eps)
        return x_t, x0, x1, t, eps

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Default training-time density: uniform on [t_min, t_max].

        Other densities enter through the ``t_sampler`` hook of
        :class:`~schnetpack.generative.losses.MatchingLoss` and
        :class:`~schnetpack.generative.transforms.Diffuse`.
        """
        span = self.t_max - self.t_min
        return self.t_min + span * torch.rand(n, device=device)

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
            "from. Pass an explicit Prior to the Sampler — one matching the "
            "statistics this coupling trained under."
        )

    # -- the Gaussian kernel: a property of the configuration -------------- #

    def gaussian_kernel_obstruction(self) -> str | None:
        """
        Why p(x_t | x0) = N(a x0, sigma^2 I) does not hold, or None if it does.

        The kernel holds iff the prior is isotropic Gaussian with a declared
        scale, the coupling pairs endpoints independently of the values and
        the schedule carries no bridge noise. Returns the first failed
        condition as a sentence for error messages.
        """
        if not self.prior.gaussian:
            return (
                f"{type(self.prior).__name__} does not declare its draws "
                "isotropic Gaussian (prior.gaussian is False)"
            )
        if self.prior.std is None:
            return (
                f"{type(self.prior).__name__} declares no scalar endpoint "
                "scale (prior.std is None)"
            )
        if not self.coupling.independent_pairs:
            return (
                f"{type(self.coupling).__name__} pairs endpoints depending "
                "on the values, so conditionally on x0 the endpoint is no "
                "longer the declared isotropic Gaussian (the marginal may "
                "survive; the one-sided kernel does not)"
            )
        # Probe gamma on interior times: None means no bridge noise by the
        # gamma contract, and a returned tensor still has to actually
        # vanish — a subclass returning zeros carries no latent.
        t = torch.linspace(self.t_min, self.t_max, 9, dtype=torch.float64)[1:-1]
        gamma = self.gamma(t)
        if gamma is not None and bool((gamma != 0).any()):
            return (
                f"{type(self).__name__} carries bridge noise (gamma != 0), "
                "so the one-sided kernel is not this process's kernel"
            )
        return None

    @property
    def has_gaussian_kernel(self) -> bool:
        """Whether the one-sided Gaussian kernel holds; see :meth:`gaussian_kernel_obstruction`."""
        return self.gaussian_kernel_obstruction() is None

    def sde(self) -> "SDE":
        """
        The (f, g) chart of this process, dx = f x dt + g dw.

        Raises:
            ValueError: if the configuration has no Gaussian kernel; the
                message names the obstruction.
        """
        # Local import: sde.py imports Process for its type and helpers.
        from schnetpack.generative.differential_equations import SDE

        return SDE(self)


class VP(Process):
    """
    Variance-preserving process (continuous-time DDPM) with linear beta.

    a = exp(-1/2 int_0^t beta), b = sqrt(1 - a^2); f = -beta/2, g^2 = beta at
    unit scale. The default unit Gaussian endpoint suits unit-variance data;
    pass ``scale`` (the data std) otherwise.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
        scale: float | None = None,
    ):
        """
        Args:
            beta_min: beta(0)
            beta_max: beta(t_max)
            t_min: smallest usable time; the score diverges as t -> 0
            t_max: largest usable time
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, shorthand for ``prior``
        """
        super().__init__(
            t_min=t_min, t_max=t_max, prior=prior, coupling=coupling, scale=scale
        )
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (t / self.t_max) * (self.beta_max - self.beta_min)

    def _log_a(self, t: torch.Tensor) -> torch.Tensor:
        int_beta = (
            self.beta_min * t
            + 0.5 * (self.beta_max - self.beta_min) * t**2 / self.t_max
        )
        return -0.5 * int_beta

    def a(self, t):
        return torch.exp(self._log_a(t))

    def a_dot(self, t):
        return -0.5 * self.beta(t) * self.a(t)

    def log_a_dot(self, t):
        # d/dt (-1/2 int beta) -- the textbook f, exact and division-free
        return -0.5 * self.beta(t)

    def b(self, t):
        # sqrt(-expm1(2 log a)) rather than sqrt(1 - a^2): at small t
        # a^2 is within an ulp of 1 and the naive form loses every digit.
        return torch.sqrt(-torch.expm1(2.0 * self._log_a(t)))

    def b_dot(self, t):
        # d/dt sqrt(1 - a^2) = -a a' / b
        a = self.a(t)
        return -a * self.a_dot(t) / torch.clamp(self.b(t), min=1e-12)


class VE(Process):
    """
    Variance-exploding process with geometric noise (score matching / SMLD).

    a = 1 and b(t) = b_min^(1 - t/t_max), so sigma(t) = b(t) * prior.std runs
    geometrically up to the prior's scale. ``VE(sigma_min, sigma_max)`` is
    the classic schedule: b_min = sigma_min / sigma_max and a Gaussian prior
    of std sigma_max, giving sigma(t) = sigma_min^(1-t) sigma_max^t. For a
    prior that owns its own scale, pass ``b_min`` and ``prior`` instead.

    sigma_max must match the data scale (rule of thumb: the largest pairwise
    distance in the dataset); a mismatch degrades samples without raising.
    A score head on this process wants ``weight=lambda t: process.b(t)**2``
    in the loss.
    """

    def __init__(
        self,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        b_min: float | None = None,
        t_min: float = 0.0,
        t_max: float = 1.0,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
        scale: float | None = None,
    ):
        """
        Args:
            sigma_min: smallest noise level of the classic schedule; give
                together with ``sigma_max``, exclusive with ``b_min``/``prior``
            sigma_max: largest noise level and the endpoint scale
            b_min: blending weight at t = 0, in (0, 1), for priors that own
                their own scale (default: 2e-4)
            t_min: smallest usable time (b(0) > 0, so 0 is fine)
            t_max: largest usable time, where b reaches 1
            prior: endpoint distribution for the ``b_min`` route (default:
                unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std for the ``b_min`` route, shorthand
                for ``prior``
        """
        if (sigma_min is None) != (sigma_max is None):
            raise TypeError(
                "Give sigma_min and sigma_max together — the schedule needs "
                "their ratio and the prior needs sigma_max."
            )
        if sigma_min is not None:
            if b_min is not None or prior is not None or scale is not None:
                raise TypeError(
                    "(sigma_min, sigma_max) already fixes b_min = "
                    "sigma_min/sigma_max and prior = GaussianPrior(sigma_max)"
                    " — pass either that pair or (b_min, prior/scale), not "
                    "both."
                )
            b_min = sigma_min / sigma_max
            prior = GaussianPrior(std=sigma_max)
        elif b_min is None:
            b_min = 2e-4
        super().__init__(
            t_min=t_min, t_max=t_max, prior=prior, coupling=coupling, scale=scale
        )
        self.b_min = b_min

    def a(self, t):
        return torch.ones_like(t)

    def a_dot(self, t):
        return torch.zeros_like(t)

    def log_a_dot(self, t):
        return torch.zeros_like(t)  # a == 1

    def b(self, t):
        return self.b_min ** (1.0 - t / self.t_max)

    def b_dot(self, t):
        return self.b(t) * math.log(1.0 / self.b_min) / self.t_max

    def t_of_sigma(self, sigma):
        # b is geometric, so t is affine in log sigma:
        #     t = t_max (1 + (log sigma - log sigma_max) / log(sigma_max/sigma_min))
        sigma = torch.as_tensor(sigma)
        log_range = -math.log(self.b_min)
        t = self.t_max * (1.0 + (torch.log(sigma) - math.log(self.std)) / log_range)
        return t.clamp(self.t_min, self.t_max)


class VELinear(Process):
    """
    Variance-exploding process with linear b: a = 1, b(t) = t / t_max.

    The Karras et al. (2022) geometry; the noise scale (their sigma_max) is
    the prior's std. See :class:`VE` for the geometric ramp.
    """

    def __init__(
        self,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
        scale: float | None = None,
    ):
        """
        Args:
            t_min: smallest usable time; b -> 0 as t -> 0
            t_max: largest usable time, where b reaches 1
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, which must match the data scale
        """
        super().__init__(
            t_min=t_min, t_max=t_max, prior=prior, coupling=coupling, scale=scale
        )

    def a(self, t):
        return torch.ones_like(t)

    def a_dot(self, t):
        return torch.zeros_like(t)

    def log_a_dot(self, t):
        return torch.zeros_like(t)  # a == 1

    def b(self, t):
        return t / self.t_max

    def b_dot(self, t):
        return torch.full_like(t, 1.0 / self.t_max)


class FlowMatching(Process):
    """
    Linear interpolant for flow matching / rectified flow: a = 1 - t, b = t.

    Pair with :class:`~schnetpack.generative.parametrizations.VelocityParametrization`
    and churn = 0. g^2 = 2t / (1 - t) diverges at t = 1, so ``t_max`` defaults
    just below 1; pure-ODE sampling (churn = 0) may use ``t_max=1.0``.
    """

    def __init__(
        self,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
        scale: float | None = None,
    ):
        """
        Args:
            t_min: smallest usable time; b -> 0 as t -> 0
            t_max: largest usable time; keep < 1 for stochastic sampling
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, shorthand for ``prior``
        """
        super().__init__(
            t_min=t_min, t_max=t_max, prior=prior, coupling=coupling, scale=scale
        )

    def a(self, t):
        return 1.0 - t

    def a_dot(self, t):
        return -torch.ones_like(t)

    def log_a_dot(self, t):
        return -1.0 / (1.0 - t)

    def b(self, t):
        return t

    def b_dot(self, t):
        return torch.ones_like(t)


class VPISSNR(Process):
    """
    Variance-preserving process with an inverse-sigmoid log-SNR
    (Kahouli et al. 2025, arXiv:2502.08598).

    TV(t) = 1 and log SNR(t) = eta * log(1/t - 1) + kappa; a, b and the
    derivatives are derived. ``eta = 2, kappa = 0`` is the variance-preserving
    sibling of :class:`FlowMatching` (same SNR schedule, TV flattened to 1).
    The log-SNR diverges at t = 0 and t = 1, so t_min and t_max sit inside.
    """

    def __init__(
        self,
        eta: float = 2.0,
        kappa: float = 0.0,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
        prior: Prior | None = None,
        coupling: Coupling | None = None,
        scale: float | None = None,
    ):
        """
        Args:
            eta: steepness of the log-SNR; > 0. eta = 2 matches flow matching
                (the paper says eta = 1 because its text defines SNR as a/b
                rather than the squared a^2/b^2 used here).
            kappa: shift of the log-SNR; > 0 moves the schedule toward
                signal, < 0 toward noise
            t_min: smallest usable time; the log-SNR diverges at t = 0
            t_max: largest usable time; the log-SNR diverges at t = 1
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, shorthand for ``prior``
        """
        super().__init__(
            t_min=t_min, t_max=t_max, prior=prior, coupling=coupling, scale=scale
        )
        self.eta = eta
        self.kappa = kappa

    def tv(self, t):
        return torch.ones_like(t)

    def log_snr(self, t):
        return self.eta * torch.log(1.0 / t - 1.0) + self.kappa

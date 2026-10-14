"""
Schedules: how fast a structure moves from the data at t = 0 to the prior.

A :class:`Schedule` defines a(t) and b(t) directly and works with any prior
(flow matching). A :class:`GaussianSchedule` defines a(t) and the noise level
sigma(t) for a Gaussian prior N(0, sigma_max^2 I) (diffusion); its b is the
normalized noise level sigma / sigma_max, so it is also a plain
:class:`Schedule`. Time derivatives default to autograd and may be overridden
analytically. Theory and the catalog: ``docs_new/generation/schedules.md``.
"""

import abc
import math
from collections.abc import Callable

import torch

__all__ = [
    "Schedule",
    "GaussianSchedule",
    "FlowMatching",
    "VP",
    "VE",
    "VELinear",
]


def _time_derivative(
    fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
) -> torch.Tensor:
    """
    d fn / dt by autograd, returned as a plain tensor without a graph.

    Assumes ``fn`` is elementwise in t (the gradient of ``fn(t).sum()`` is
    then the per-sample derivative). Works under ``torch.no_grad()``. A
    schedule with learnable parameters gets no gradient through this and
    should override the derivative analytically.
    """
    with torch.enable_grad():
        t_ = t.detach().clone().requires_grad_(True)
        y = fn(t_)
        if not y.requires_grad:
            # constant schedule: no graph at all
            return torch.zeros_like(t)
        (grad,) = torch.autograd.grad(y.sum(), t_, allow_unused=True)
    return torch.zeros_like(t) if grad is None else grad


class Schedule(abc.ABC):
    """
    Schedule x_t = a(t) x0 + b(t) x1 on [t_min, t_max], for any prior.

    a(0) = 1, b(0) = 0, b(1) = 1, and a b' - a' b > 0 so the structure gets
    noisier with t.
    """

    def __init__(self, t_min: float, t_max: float):
        """
        Args:
            t_min: smallest usable path time
            t_max: largest usable path time, where the prior is drawn
        """
        self.t_min = t_min
        self.t_max = t_max

    @abc.abstractmethod
    def a(self, t: torch.Tensor) -> torch.Tensor:
        """Data coefficient a(t), shaped like t."""
        raise NotImplementedError

    @abc.abstractmethod
    def b(self, t: torch.Tensor) -> torch.Tensor:
        """Prior coefficient b(t), shaped like t."""
        raise NotImplementedError

    def a_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`a` (autograd unless overridden)."""
        return _time_derivative(self.a, t)

    def b_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`b` (autograd unless overridden)."""
        return _time_derivative(self.b, t)

    def log_a_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log a(t) = a_dot / a, the drift f of the SDE chart.

        Override with the closed form where one exists: it is usually
        division-free and stays finite where a underflows to zero.
        """
        return _time_derivative(lambda s: torch.log(self.a(s)), t)


class GaussianSchedule(Schedule):
    """
    Schedule x_t = a(t) x0 + sigma(t) eps for a Gaussian prior N(0, sigma_max^2 I).

    A subclass defines (:meth:`a`, :meth:`sigma`); b = sigma / sigma_max is
    derived.
    """

    def __init__(self, sigma_max: float, t_min: float, t_max: float):
        """
        Args:
            sigma_max: noise level at t = 1, the std of the prior
            t_min: smallest usable path time
            t_max: largest usable path time, where the prior is drawn
        """
        super().__init__(t_min=t_min, t_max=t_max)
        self.sigma_max = sigma_max

    @abc.abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise level sigma(t), shaped like t."""
        raise NotImplementedError

    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`sigma` (autograd unless overridden)."""
        return _time_derivative(self.sigma, t)

    def b(self, t: torch.Tensor) -> torch.Tensor:
        """Normalized noise level sigma(t) / sigma_max."""
        return self.sigma(t) / self.sigma_max

    def b_dot(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_dot(t) / self.sigma_max

    def t_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Inverse of :meth:`sigma`: the path time at which a noise level sits.

        Bisection over [t_min, t_max], which only needs sigma to be monotone;
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
        # sigma is monotone but its direction is the schedule's business: read
        # it off the endpoints rather than assuming noise grows with t.
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


class FlowMatching(Schedule):
    """
    Linear schedule of flow matching / rectified flow: a = 1 - t, b = t.

    With a Gaussian prior, g^2 = 2t / (1 - t) diverges at t = 1, so ``t_max``
    defaults just below 1; pure-ODE sampling (eta2 = 0) may use
    ``t_max=1.0``.
    """

    def __init__(self, t_min: float = 1e-3, t_max: float = 1.0 - 1e-3):
        """
        Args:
            t_min: smallest usable time; b -> 0 as t -> 0
            t_max: largest usable time; keep < 1 for stochastic sampling
        """
        super().__init__(t_min=t_min, t_max=t_max)

    def a(self, t):
        return 1.0 - t

    def b(self, t):
        return t

    def a_dot(self, t):
        return -torch.ones_like(t)

    def b_dot(self, t):
        return torch.ones_like(t)

    def log_a_dot(self, t):
        return -1.0 / (1.0 - t)


class VP(GaussianSchedule):
    """
    Variance-preserving schedule (continuous-time DDPM) with linear beta.

    a = exp(-1/2 int_0^t beta), sigma = sqrt(1 - a^2), sigma_max = 1;
    f = -beta/2, g^2 = beta. Suits unit-variance data.
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
        super().__init__(sigma_max=1.0, t_min=t_min, t_max=t_max)
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

    def sigma(self, t):
        # sqrt(-expm1(2 log a)) rather than sqrt(1 - a^2): at small t
        # a^2 is within an ulp of 1 and the naive form loses every digit.
        return torch.sqrt(-torch.expm1(2.0 * self._log_a(t)))

    def sigma_dot(self, t):
        # d/dt sqrt(1 - a^2) = -a a' / sigma
        a = self.a(t)
        return -a * self.a_dot(t) / torch.clamp(self.sigma(t), min=1e-12)


class VE(GaussianSchedule):
    """
    Variance-exploding schedule with geometric noise (score matching / SMLD).

    a = 1, sigma(t) = sigma_min^(1 - t/t_max) sigma_max^(t/t_max), so
    b(t) = (sigma_min / sigma_max)^(1 - t/t_max).

    sigma_max must match the data scale (rule of thumb: the largest pairwise
    distance in the dataset); a mismatch degrades samples without raising.
    A score head on this schedule wants ``weight=lambda t: noising.b(t)**2``
    in the loss.
    """

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        """
        Args:
            sigma_min: noise level at t = 0
            sigma_max: noise level at t_max, and the prior's std
            t_min: smallest usable time (sigma(0) > 0, so 0 is fine)
            t_max: largest usable time
        """
        super().__init__(sigma_max=sigma_max, t_min=t_min, t_max=t_max)
        self.sigma_min = sigma_min

    def a(self, t):
        return torch.ones_like(t)

    def a_dot(self, t):
        return torch.zeros_like(t)

    def log_a_dot(self, t):
        return torch.zeros_like(t)  # a == 1

    def sigma(self, t):
        return self.sigma_max * (self.sigma_min / self.sigma_max) ** (
            1.0 - t / self.t_max
        )

    def sigma_dot(self, t):
        return self.sigma(t) * math.log(self.sigma_max / self.sigma_min) / self.t_max

    def t_of_sigma(self, sigma):
        # sigma is geometric, so t is affine in log sigma:
        #     t = t_max (1 + (log sigma - log sigma_max) / log(sigma_max/sigma_min))
        sigma = torch.as_tensor(sigma)
        log_range = math.log(self.sigma_max / self.sigma_min)
        t = self.t_max * (
            1.0 + (torch.log(sigma) - math.log(self.sigma_max)) / log_range
        )
        return t.clamp(self.t_min, self.t_max)


class VELinear(GaussianSchedule):
    """
    Variance-exploding schedule with linear noise: a = 1,
    sigma(t) = sigma_max t / t_max (the EDM geometry of Karras et al. 2022).
    See :class:`VE` for the geometric ramp.
    """

    def __init__(
        self,
        sigma_max: float = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ):
        """
        Args:
            sigma_max: noise level at t_max, and the prior's std
            t_min: smallest usable time; sigma -> 0 as t -> 0
            t_max: largest usable time, where sigma reaches sigma_max
        """
        super().__init__(sigma_max=sigma_max, t_min=t_min, t_max=t_max)

    def a(self, t):
        return torch.ones_like(t)

    def a_dot(self, t):
        return torch.zeros_like(t)

    def log_a_dot(self, t):
        return torch.zeros_like(t)  # a == 1

    def sigma(self, t):
        return self.sigma_max * t / self.t_max

    def sigma_dot(self, t):
        return torch.full_like(t, self.sigma_max / self.t_max)

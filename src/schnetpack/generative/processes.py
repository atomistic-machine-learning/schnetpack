"""
Forward processes — schedule, prior and coupling in one object.

A process is the noising machine of a generative model: the interpolant

    x_t = a(t) x0 + b(t) x1 + gamma(t) eps,

with x0 the data at t = 0 and x1 the prior endpoint at t = t_max, together
with what that endpoint *is* and how endpoint batches are paired. The bridge
noise gamma is identically zero for every process here (see
:meth:`Process.gamma`), so the familiar two-term form is what actually runs.

One class carries three roles, still factored internally:

- **The schedule** — the subclass. a and b are dimensionless blending
  weights: b runs inside [0, 1] and reaches 1 at t_max, so the schedule
  says *when* the endpoint takes over, never *how large* it is. A subclass
  defines either (a, b) or (tv, log_snr) — the total variance a^2 + b^2 and
  the log signal-to-noise ratio log(a^2 / b^2), the TV/SNR reparametrization
  of Kahouli et al. (2025), arXiv:2502.08598. Each pair derives the other,
  time derivatives default to autograd, and everything else — the forward
  SDE drift f = a'/a, the diffusion g^2, the closed-form marginals that make
  simulation-free training possible — follows. :class:`VP`, :class:`VE` and
  :class:`FlowMatching` are subclasses whose whole content is a schedule,
  spelled in the literature's vocabulary.
- **The endpoint** — the prior (:mod:`schnetpack.generative.priors`), which
  owns what x1 is, including its scale: the noise level of the process is
  sigma(t) = b(t) * prior.std, and ``VE(sigma_min=0.3, sigma_max=30.0)``
  puts sigma_max on the prior it builds internally. The scale is declared
  once and never mirrored.
- **The pairing** — the coupling (:mod:`schnetpack.generative.couplings`),
  which re-pairs already-drawn batches and never draws.

:meth:`Process.perturb` composes the three, one owner per line::

    x1 = prior.sample_like(x0, context)   # what x1 is       (the prior)
    x0, x1 = coupling.pair(x0, x1)        # how paired       (the coupling)
    x_t = self.interpolate(x0, x1, t)     # when it takes over (the schedule)

and owns every random draw of the forward side: for bridge schedules the
training target must see the *same* eps that entered the interpolant, so
perturb draws it and returns it.

**The Gaussian kernel is a property, not a class.** The score and noise
training targets are statements about the one-sided kernel
p(x_t | x0) = N(a x0, sigma^2 I), which holds exactly when the prior is an
isotropic Gaussian of declared scale, the coupling preserves x1's marginal,
and the schedule carries no bridge noise.
:meth:`Process.gaussian_kernel_obstruction` checks those conditions against
the actual configuration and names the first that fails;
:attr:`Process.has_gaussian_kernel` is the boolean. It gates the score/noise
parametrizations (checked in their ``validate``, called by every consumer
constructor) and the Gaussian-only closed forms — the perturbation
:meth:`Process.kernel` and the exact :meth:`Process.posterior` that
ancestral sampling and DDIM discretize. Judging the configuration rather
than the class is what lets one schedule serve both modes: the same
:class:`VE` is a Gaussian diffusion under a ``GaussianPrior`` and a general
stochastic interpolant (Albergo et al., arXiv:2303.08797) under a shape
prior, with no second hierarchy.

What the network predicts — the training targets and the conversions
between score, noise, denoiser and velocity — lives on
:mod:`schnetpack.generative.parametrizations`, which reads everything it
needs from the process. Keeping that split means a new parametrization
never touches this file, and a new schedule never touches that one.

Why the interpolant is the primitive rather than (f, g): training needs x_t
for a random (x0, t) in one shot, which the interpolant gives and the SDE
would make you integrate. And (f, g) derive from it by differentiating the
schedule, whereas the converse costs an ODE solve per schedule — and for
flow matching there is no intrinsic g to start from, since its diffusion is
a sampler choice.

Times may be per-sample (shape ``(n_samples,)``): samples in one batch can
sit at different times, which per-sample training and adaptive integrators
rely on.
"""

import abc
import inspect
import math
from typing import Callable, Optional, Tuple

import torch

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
    Right-pad a per-sample time tensor for broadcasting against sample data.

    Args:
        t: scalar (0-dim) or per-sample tensor of shape (n_samples,)
        x: data tensor of shape (n_samples, ...)
    """
    if t.dim() == 0:
        return t
    return t.reshape(t.shape[0], *([1] * (x.dim() - 1)))


class Process(abc.ABC):
    """
    Forward process x_t = a(t) x0 + b(t) x1 + gamma(t) eps on [t_min, t_max],
    with x1 ~ prior and (x0, x1) matched by the coupling.

    Convention: b is a normalized blending weight, 0 <= b <= 1 with
    b(t_max) = 1 (exactly, or within the t_max approximation). A schedule
    that put a physical scale into b would be smuggling the endpoint's law
    into the geometry — that scale belongs to the prior drawing x1.

    A subclass fixes the schedule by defining **either** of two pairs:

    - :meth:`a` and :meth:`b` — the coefficients directly;
    - :meth:`tv` and :meth:`log_snr` — the total variance a^2 + b^2 and
      the log signal-to-noise ratio log(a^2 / b^2).

    Whichever pair is given, the other is derived (see :meth:`a`), so both
    are always available. Defining neither is a TypeError at class-definition
    time — the two routes are mutually recursive, and this is where that
    shows up as a readable error rather than a RecursionError at the first
    call.

    The TV/SNR route is the reparametrization of Kahouli et al. (2025),
    "Total-Variance/Signal-to-Noise-Ratio Disentangled Diffusion"
    (arXiv:2502.08598). The point is that the two knobs are independent: TV
    fixes how large x_t is, SNR fixes how much of it is signal, and neither
    constrains the other. Written as a/b those choices are tangled —
    changing the noise level moves the total variance too — which is why the
    schedules that work are folklore. See :class:`VPISSNR`.

    The derivatives :meth:`a_dot` and :meth:`b_dot` default to autograd
    through the schedule, so a subclass need not supply them. Overriding
    them with analytic forms is a speed and precision optimization, not a
    requirement; every schedule in this module does, and their agreement
    with autograd is tested.

    ``t_min`` and ``t_max`` bound the usable time range and are consumed by
    prior sampling, the default training time sampler and the sampler grids.
    They exist because the endpoints are where schedules misbehave: the
    score diverges as b -> 0, and some diffusion coefficients blow up at
    t_max.
    """

    def __init__(
        self,
        t_min: float,
        t_max: float,
        prior: Optional[Prior] = None,
        coupling: Optional[Coupling] = None,
        scale: Optional[float] = None,
    ):
        """
        Args:
            t_min: smallest usable path time
            t_max: largest usable path time (where the prior is drawn)
            prior: distribution of the x1 endpoint, for training draws and
                (via :meth:`sampling_prior`) the sampling start; mutually
                exclusive with ``scale``
            coupling: how endpoint batches are paired (default: identity,
                i.e. every sample keeps its fresh draw)
            scale: standard deviation of a Gaussian endpoint — sugar for
                ``prior=GaussianPrior(std=scale)``, default 1. This is the
                physical size the dimensionless schedule lacks: b(t) says
                *when* the endpoint takes over, sigma(t) = b(t) * scale says
                how big it is.
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
        """
        Reject a subclass that defines neither schedule pair.

        :meth:`a`/:meth:`b` and :meth:`tv`/:meth:`log_snr` are defined
        in terms of each other, so a subclass supplying neither would recurse
        until the stack ran out — on first use, far from the cause. Catching
        it here turns that into a TypeError naming the class that is wrong.
        """
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
        """
        Data coefficient a(t), shaped like t.

        Derived from the TV/SNR pair unless overridden. Inverting
        TV = a^2 + b^2 and SNR = a^2 / b^2 gives

            a^2 = TV * SNR / (1 + SNR) = TV * sigmoid(log SNR),

        using SNR / (1 + SNR) = sigmoid(log SNR). Going through the sigmoid
        is not cosmetic: SNR itself spans the whole positive axis and
        overflows a float at the ends of a schedule, where sigmoid(log SNR)
        simply saturates at 1.

        Assumes a >= 0, which the square root cannot recover on its own.
        """
        return torch.sqrt(self.tv(t) * torch.sigmoid(self.log_snr(t)))

    def b(self, t: torch.Tensor) -> torch.Tensor:
        """
        Noise coefficient b(t), shaped like t.

        The mirror of :meth:`a`: b^2 = TV / (1 + SNR) =
        TV * sigmoid(-log SNR).
        """
        return torch.sqrt(self.tv(t) * torch.sigmoid(-self.log_snr(t)))

    def tv(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total variance a^2 + b^2, shaped like t.

        The variance of x_t for unit-variance data and noise: how big the
        interpolated sample is, independent of how much of it is signal. A
        variance-preserving schedule is exactly one with tv == 1.

        Derived from a/b unless overridden.
        """
        return self.a(t) ** 2 + self.b(t) ** 2

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Log signal-to-noise ratio, 2 (log a - log b).

        The other half of the TV/SNR pair, and the one to define: it is
        finite and well conditioned over the whole schedule where
        :meth:`snr` itself overflows. If you have the ratio rather than its
        log, return ``torch.log(gamma(t))``.

        Derived from a/b unless overridden. Note the convention:
        SNR = a^2 / b^2 is the *squared* ratio, matching the gamma of
        the TV/SNR reference implementation (whose ``log_gamma`` is this
        function) and Kingma's log-SNR.
        """
        return 2.0 * (torch.log(self.a(t)) - torch.log(self.b(t)))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio a^2 / b^2."""
        return self.a(t) ** 2 / self.b(t) ** 2

    # -- derivatives ------------------------------------------------------ #

    def a_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`a`, shaped like t (autograd default)."""
        return self._time_derivative(self.a, t)

    def b_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of :meth:`b`, shaped like t (autograd default)."""
        return self._time_derivative(self.b, t)

    def log_a_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log a(t), shaped like t. Equals a_dot / a.

        This is the hook the identity buys, and :meth:`f` is exactly it. The
        default here is honest about its limits: autograd through ``log(a)``
        applies the chain rule as (1 / a) * a_dot, so it *is* the
        quotient and degenerates in all the same places. What it gives you
        is a place to override.

        Override it whenever log a is known in closed form, because that
        form is usually the one with no division in it — VP's is simply
        -beta/2, finite everywhere including where a underflows to zero and
        the quotient becomes 0/0. Every schedule in this module does so.
        """
        return self._log_derivative(self.a, t)

    def log_b_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt log b(t). Equals b_dot / b; see :meth:`log_a_dot`."""
        return self._log_derivative(self.b, t)

    def log_snr_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        d/dt log SNR(t), shaped like t.

        Differentiates :meth:`log_snr` directly rather than differencing
        :meth:`log_a_dot` and :meth:`log_b_dot`: one pass instead of
        two, and — for a schedule defined the TV/SNR way — the derivative of
        the closed form the subclass actually wrote, with no a or b formed
        along the way and so no quotient to degenerate. That is what makes
        :meth:`g2` well behaved on schedules whose a reaches zero.

        Non-positive for any sensible schedule — signal only ever turns into
        noise — which is what makes :meth:`g2` non-negative.
        """
        return self._time_derivative(self.log_snr, t)

    @staticmethod
    def _log_derivative(
        fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """d/dt log fn(t), by autograd through the log rather than by dividing."""
        return Process._time_derivative(lambda s: torch.log(fn(s)), t)

    @staticmethod
    def _time_derivative(
        fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """
        d fn / dt by autograd, so a schedule need not be differentiated by hand.

        Differentiates ``fn(t).sum()``, which gives the per-sample derivative
        only because schedules are elementwise in t — sample i's value does
        not depend on sample j's time, so the sum's gradient is the vector of
        individual derivatives. A schedule coupling times across the batch
        would silently get the wrong answer here and must override.

        ``enable_grad`` because this runs inside ``torch.no_grad()`` during
        sampling, where the graph would otherwise never be built.

        The result is a plain tensor: ``t`` is detached going in and
        ``create_graph`` is False, so nothing here carries a graph back out.
        That matters beyond tidiness — :class:`~schnetpack.generative.transforms.Diffuse`
        builds targets inside the dataloader, and a target holding a grad_fn
        cannot be pickled to a worker process. The cost is that a schedule
        with *learnable* parameters gets no gradient through this; such a
        schedule should override the derivative analytically.

        A constant schedule (``a = ones_like(t)``) produces a tensor with no
        grad_fn at all, which autograd reports as unused rather than as zero
        — hence the two zero fallbacks.
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
        Bridge processes (Schrödinger bridge, and any interpolant whose x_t
        depends on both endpoints plus its own noise) override this; nothing
        else needs to.

        Not to be confused with the gamma of the TV/SNR literature, which is
        the signal-to-noise ratio and lives here as :meth:`snr` /
        :meth:`log_snr`. This name predates that and means the third
        interpolant coefficient.
        """
        return None

    # -- derived scalars -------------------------------------------------- #

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient f(t) = a'/a of the forward SDE.

        Which is d/dt log a — see :meth:`log_a_dot`, where the quotient
        is avoided rather than computed.
        """
        return self.log_a_dot(t)

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion of the forward SDE, g^2 = -sigma^2 d/dt log SNR.

        The usual unit-scale form is g^2 = 2 b b' - 2 f b^2. Factoring b^2
        out of both terms leaves the two log-derivatives,

            g^2 = 2 b^2 (d/dt log b - d/dt log a),

        and log SNR = 2 (log a - log b) makes that bracket exactly
        -1/2 d/dt log SNR. So the whole diffusion is one log-derivative of
        one schedule — no quotient, and nothing that degenerates where a or
        b vanish. It also puts the sign where it can be read: g^2 >= 0
        precisely because SNR decreases. A constant endpoint scale
        multiplies the noise level everywhere and leaves d/dt log SNR
        unchanged, which is why the scale enters simply as sigma^2 in place
        of b^2.

        This is the diffusion that shares the process's marginals. It is a
        *canonical* choice rather than an intrinsic property of the
        interpolant: any g gives the same marginals as long as the drift
        matches, and the sampler picks how much of it to use via its churn
        knob. Flow matching runs at churn = 0 and never touches this.
        """
        return -self.sigma(t) ** 2 * self.log_snr_dot(t)

    def a_b(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both marginal coefficients at once, each shaped like t."""
        return self.a(t), self.b(t)

    # -- scale: the prior's declaration, exposed once ---------------------- #

    @property
    def std(self) -> float:
        """
        Endpoint scale — ``prior.std``, with a diagnosis when undeclared.

        Consumers read the scale here rather than from ``prior.std``
        directly, so a prior without a scalar scale fails with this error
        instead of a ``NoneType`` arithmetic error at the first use.
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
        """
        Noise level of the process, sigma(t) = b(t) * prior.std.

        The schedule's b is a dimensionless blending weight; the prior's std
        is the endpoint's physical size. Their product is the sigma(t) of
        the diffusion literature — for ``VE(0.3, 30.0)`` this is exactly
        sigma_min^(1-t) sigma_max^t.
        """
        return self.b(t) * self.std

    def t_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        The inverse of :meth:`sigma` — where on the path a noise level sits.

        Everything that reasons in noise levels rather than times needs this:
        a training density stated in sigma
        (:class:`~schnetpack.generative.times.LogNormalSigmaTimes`), a
        sigma-spaced sampling grid (Karras/EDM). They speak sigma because
        that is the physical quantity, while the rest of the library speaks
        t, and only the process knows the map between them.

        Solved by bisection on the whole schedule, which needs no more than
        b's monotonicity — true of every shipped schedule, since a b that
        turned around would revisit the same noise level twice and the
        inverse would not be a function. Subclasses with a closed form
        override this (see :meth:`VE.t_of_sigma`); the fallback exists so a
        new schedule gets the capability for free, not because bisection is
        the intended route.

        Args:
            sigma: noise levels, any shape

        Returns:
            Times of the same shape, clamped to [t_min, t_max] — a sigma
            outside the schedule's range maps to the nearest endpoint rather
            than raising, so a density with tails wider than the schedule
            piles them on the ends instead of failing mid-epoch.
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
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Place a sample on the path: x_t = a x0 + b x1 (+ gamma eps).

        Args:
            x0: data endpoint, shape (n_samples, ...)
            x1: prior endpoint, shape of x0; Gaussian noise for the
                independent coupling, a paired endpoint for bridges
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
        x1: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        context=None,
        groups: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Draw endpoints, pair them, place them on the path.

        Owns every random draw of the forward side. That matters for the
        bridge noise in particular: the training target must see the *same*
        eps that entered the interpolant, so it is drawn here and returned
        rather than hidden inside :meth:`interpolate`.

        Args:
            x0: data batch, shape (n_samples, ...)
            x1: endpoint batch to use instead of drawing from the prior;
                still passed through the coupling
            t: path times, per-sample or scalar; drawn from
                :meth:`sample_t` if not given
            context: generation-time conditioning handed to the prior
            groups: labels restricting which rows the coupling may exchange
                endpoints between — for a collated batch of molecules,
                ``(idx_m, Z)``, so the re-pairing stays inside one molecule and
                one element. Passed straight to
                :meth:`~schnetpack.generative.couplings.Coupling.pair`; without
                it a batch is one unrestricted point cloud.

        Returns:
            (x_t, x0, x1, t, eps) — the perturbed batch, the (possibly
            re-paired) endpoints, the times, and the bridge noise (None for
            gamma = 0 schedules).
        """
        if x1 is None:
            x1 = self.prior.sample_like(x0, context)
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

    def sample_t(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Training-time distribution p(t): uniform on [t_min, t_max].

        Stops short of t = 0 because the score target diverges there; the
        noise and denoiser targets are well behaved at 0, so a process
        trained on those may widen the range via the ``t_sampler`` hooks on
        :class:`~schnetpack.generative.losses.MatchingLoss` and
        :class:`~schnetpack.generative.transforms.Diffuse` — the same hook
        that admits the EDM/GPFF log-normal-sigma density.
        """
        span = self.t_max - self.t_min
        return self.t_min + span * torch.rand(n, device=device)

    # -- sampling ----------------------------------------------------------#

    def sampling_prior(self) -> Prior:
        """
        Start distribution for the reverse process.

        With b(t_max) = 1 the state the reverse process must start from is
        x1's marginal. A coupling that only re-pairs leaves that marginal
        untouched, so the start is the *training prior itself* — the same
        object, not a restatement. A marginal-changing coupling has no
        data-free start distribution, and the error demands an explicit
        :class:`~schnetpack.generative.priors.Prior` instead of guessing.
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

    def gaussian_kernel_obstruction(self) -> Optional[str]:
        """
        Why the one-sided Gaussian kernel does not hold — or None if it does.

        The kernel p(x_t | x0) = N(a x0, sigma^2 I) is exact iff the prior
        is an isotropic Gaussian with a declared scale, the coupling at most
        re-orders x1 across the batch (re-pairing exchangeable draws leaves
        the marginal Gaussian), and the schedule carries no bridge noise (a
        Gaussian endpoint plus gamma is still Gaussian, but then
        sigma^2 = b^2 std^2 + gamma^2 — fold it into the noise coefficient
        first). Those are exactly the assumptions behind the score/noise
        training targets and the closed forms :meth:`kernel` and
        :meth:`posterior`.

        Returns the first failed condition as a readable sentence, so the
        callers that must refuse — the score/noise parametrizations'
        ``validate``, the closed forms — can say *why*.
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
        if not self.coupling.preserves_marginal:
            return (
                f"{type(self.coupling).__name__} reshapes x1's marginal "
                "from the data, so the endpoint is no longer the declared "
                "isotropic Gaussian"
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
        """
        Whether p(x_t | x0) = N(a x0, sigma^2 I) holds for this
        configuration — see :meth:`gaussian_kernel_obstruction`.
        """
        return self.gaussian_kernel_obstruction() is None

    def _require_gaussian_kernel(self) -> None:
        obstruction = self.gaussian_kernel_obstruction()
        if obstruction is not None:
            raise ValueError(
                f"This closed form is a statement about the Gaussian kernel "
                f"p(x_t | x0) = N(a x0, sigma^2 I), which this process does "
                f"not have: {obstruction}."
            )

    # -- Gaussian-only closed forms ----------------------------------------#

    def kernel(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean coefficient and std of the perturbation kernel p(x_t | x0),
        i.e. (a(t), sigma(t)) with p(x_t | x0) = N(a x0, sigma^2 I).

        Raises unless :attr:`has_gaussian_kernel`.
        """
        self._require_gaussian_kernel()
        return self.a(t), self.sigma(t)

    def posterior(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and std of the exact Gaussian posterior p(x_s | x_t, x0), s < t.

        The linear-Gaussian Markov structure gives x_t | x_s ~
        N((a_t/a_s) x_s, sigma_t^2 - (a_t/a_s)^2 sigma_s^2), and
        conditioning the joint on (x_t, x0) yields

            mean = (r sigma_s^2 / sigma_t^2) x_t + (a_s var_ts / sigma_t^2) x0,
            var  = sigma_s^2 var_ts / sigma_t^2,

        with r = a_t/a_s and var_ts = sigma_t^2 - r^2 sigma_s^2. For VP at
        the DDPM discretization this is the textbook ancestral posterior;
        for VE (a = 1) it reduces to the familiar
        mean = (sigma_s^2/sigma_t^2) x_t + (1 - sigma_s^2/sigma_t^2) x0.
        This is what ancestral sampling and DDIM discretize — with a known
        x0 (or a model's x0-estimate in its place) the step is exact, no
        score reconstruction involved.

        Raises unless :attr:`has_gaussian_kernel`.

        Args:
            x_t: states at time t
            x0: clean data (or its estimate)
            t: current times, per-sample or scalar
            s: target times, s < t elementwise

        Returns:
            (mean, std): mean shaped like x_t, std shaped like t.
        """
        a_t, sig_t = self.kernel(t)
        a_s, sig_s = self.kernel(s)
        r = a_t / a_s
        var_ts = sig_t**2 - r**2 * sig_s**2
        mean = (
            expand_t(r * sig_s**2 / sig_t**2, x_t) * x_t
            + expand_t(a_s * var_ts / sig_t**2, x0) * x0
        )
        std = torch.sqrt(torch.clamp(sig_s**2 * var_ts / sig_t**2, min=0.0))
        return mean, std


class VP(Process):
    """
    Variance-preserving process (continuous-time DDPM) with linear beta.

    a = exp(-1/2 int_0^t beta), b = sqrt(1 - a^2), which gives
    f = -1/2 beta and g^2 = beta (at unit scale). Unit-variance data keeps
    unit variance for all t, and p_{t_max} -> N(0, I). The default unit
    Gaussian endpoint is right for unit-variance data; pass ``scale`` (the
    data std) for unnormalized data.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        prior: Optional[Prior] = None,
        coupling: Optional[Coupling] = None,
        scale: Optional[float] = None,
    ):
        """
        Args:
            beta_min: beta(0)
            beta_max: beta(t_max)
            t_min: smallest usable time; the score diverges as t -> 0
            t_max: largest usable time
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, sugar for ``prior``; the data std
                for unnormalized data
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

    a = 1 and b(t) = b_min^(1 - t/t_max): b climbs geometrically from b_min
    at t = 0 to exactly 1 at t_max, so the mean never moves and
    x_{t_max} = x1 — whatever endpoint the prior drew. The classic
    (sigma_min, sigma_max) schedule is this process with the scale on the
    prior, a split the constructor performs once, internally::

        VE(sigma_min=0.3, sigma_max=30.0)
        # == b_min = sigma_min / sigma_max, prior = GaussianPrior(sigma_max)

    giving sigma(t) = sigma_min^(1-t) sigma_max^t exactly. For a structured
    endpoint (a GPFF shape prior, a scaffold), pass the dimensionless
    ``b_min`` and the prior instead::

        VE(b_min=1e-2, prior=my_shape_prior)

    Since b(0) = b_min > 0, nothing is singular at t = 0 and t_min may stay
    there.

    The famous VE footgun lives on the *prior*: unlike VP, the process is
    not scale-free, and sigma_max has to match your data. The rule of thumb
    (Song & Ermon 2020) is the largest pairwise distance in the dataset —
    enough to drown out the data, and no more. Overshooting does not fail
    loudly; a model trained there samples badly while its loss looks fine.
    If sampling produces garbage, check sigma_max first. The dimensionless
    b_min, i.e. sigma_min/sigma_max, is only the *dynamic range*: how far
    below the endpoint scale the noise starts.

    Predicting the score directly here also wants ``weight=lambda t:
    process.b(t)**2`` in the loss: b spans orders of magnitude, so the score
    target does too, and an unweighted L2 sees only its low-noise end. That
    weighting makes the objective identical to eps-matching (up to the
    constant endpoint scale), which is the other way to get the same result.
    """

    def __init__(
        self,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        b_min: Optional[float] = None,
        t_min: float = 0.0,
        t_max: float = 1.0,
        prior: Optional[Prior] = None,
        coupling: Optional[Coupling] = None,
        scale: Optional[float] = None,
    ):
        """
        Args:
            sigma_min: smallest noise level of the classic schedule; give
                together with ``sigma_max``, exclusive with ``b_min``/``prior``
            sigma_max: largest noise level, and the endpoint scale — must
                match the data scale (largest pairwise distance rule)
            b_min: blending weight at t = 0, i.e. sigma_min/sigma_max of the
                classic schedule, in (0, 1); the dimensionless route, for
                priors that own their own scale (default: 2e-4)
            t_min: smallest usable time
            t_max: largest usable time (where b reaches 1)
            prior: endpoint distribution for the ``b_min`` route (default:
                unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std for the ``b_min`` route, sugar for
                ``prior`` — the sigma_max of the equivalent classic schedule
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
        # b is geometric, so t is *affine in log sigma* — invert in closed
        # form rather than bisecting. Writing L = log(sigma_max/sigma_min),
        # sigma(t) = sigma_max b_min^(1 - t/t_max) gives
        #     t = t_max (1 + (log sigma - log sigma_max) / L).
        # The affine shape is what makes a log-normal density over sigma a
        # plain normal over t (see LogNormalSigmaTimes).
        sigma = torch.as_tensor(sigma)
        log_range = -math.log(self.b_min)
        t = self.t_max * (1.0 + (torch.log(sigma) - math.log(self.std)) / log_range)
        return t.clamp(self.t_min, self.t_max)


class VELinear(Process):
    """
    Variance-exploding process with linear b: a = 1, b(t) = t / t_max.

    The Karras et al. (2022) geometry — noise growing linearly to the
    endpoint — in the shared normalized convention: b runs 0 -> 1 and the
    noise scale (their sigma_max) is the ``scale``. Prefer :class:`VE` when
    you want the geometric ramp of classic score matching; this one is the
    straight ramp.
    """

    def __init__(
        self,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        prior: Optional[Prior] = None,
        coupling: Optional[Coupling] = None,
        scale: Optional[float] = None,
    ):
        """
        Args:
            t_min: smallest usable time; b -> 0 as t -> 0
            t_max: largest usable time (where b reaches 1)
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std — the sigma_max of the Karras
                schedule, which must match the data scale
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

    The velocity target is constant along a pair, x1 - x0, which is what
    makes the learned field straight and few-step sampling work. Pair with a
    velocity parametrization and churn = 0.

    The prior at t = 1 is exact here (a(1) = 0, b(1) = 1), but g^2 =
    2 t / (1 - t) diverges there, so ``t_max`` defaults just below 1. That
    costs a dropped a(t_max) x0 term of order 1e-3 times the data scale at
    the start of sampling — negligible, but it is why the default is not
    exactly 1. Pure-ODE users (churn = 0, where g^2 is never evaluated) can
    pass ``t_max=1.0`` and start from the exact endpoint.
    """

    def __init__(
        self,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
        prior: Optional[Prior] = None,
        coupling: Optional[Coupling] = None,
        scale: Optional[float] = None,
    ):
        """
        Args:
            t_min: smallest usable time; b -> 0 as t -> 0
            t_max: largest usable time; keep < 1 for stochastic sampling
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, sugar for ``prior``
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
    (arXiv:2502.08598).

    The headline schedule of the TV/SNR paper, and the one process here
    defined the TV/SNR way rather than the a/b way::

        TV(t)      = 1                          (variance preserving)
        log SNR(t) = eta * log(1/t - 1) + kappa

    a and b follow from :meth:`Process.a`, and the derivatives from
    autograd — nothing else is written. That is the point of the class: it
    is the whole schedule, and it exists to show that the second route is a
    real one rather than a convenience.

    Writing it this way is what makes the two knobs mean something
    separately. ``eta`` sets how fast signal turns into noise and ``kappa``
    shifts where the schedule sits, while TV = 1 pins the total variance no
    matter what either does. In a/b coordinates that separation does not
    exist: any change to the noise level moves the total variance too, and
    you have to fix it back up by hand.

    log(1/t - 1) is the inverse of the sigmoid, so a^2 = sigmoid(log SNR)
    runs smoothly from ~1 to ~0 with no endpoint chosen by fiat — but it
    also diverges at t = 0 and t = 1, which is why ``t_min`` and ``t_max``
    sit strictly inside.

    ``eta = 2`` is the natural default: with the paper's companion TV
    schedule ``(1-t)^eta + t^eta exp(-kappa)`` the same inversion gives
    a^2 = (1-t)^eta and b^2 = t^eta exp(-kappa), so eta = 2, kappa = 0 is
    exactly optimal-transport flow matching (:class:`FlowMatching`). This
    class is its variance-preserving sibling: same SNR schedule, TV
    flattened to 1.

    (The paper quotes eta = 1 for that correspondence because its text
    defines SNR as a/b, while its code — and :meth:`Process.log_snr` — uses
    the squared a^2/b^2. The factor of two lands on eta.)
    """

    def __init__(
        self,
        eta: float = 2.0,
        kappa: float = 0.0,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
        prior: Optional[Prior] = None,
        coupling: Optional[Coupling] = None,
        scale: Optional[float] = None,
    ):
        """
        Args:
            eta: steepness of the log-SNR; > 0. eta = 2 matches flow matching.
            kappa: shift of the log-SNR; > 0 moves the schedule toward
                signal, < 0 toward noise
            t_min: smallest usable time; the log-SNR diverges at t = 0
            t_max: largest usable time; the log-SNR diverges at t = 1
            prior: endpoint distribution (default: unit Gaussian)
            coupling: pairing rule (default: identity)
            scale: Gaussian endpoint std, sugar for ``prior``
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

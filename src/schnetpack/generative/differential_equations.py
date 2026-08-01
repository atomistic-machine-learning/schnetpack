"""
The differential-equation representations of a process, and their reversal.

A process is defined by its interpolant, x_t = a(t) x0 + b(t) x1 — that
chart always exists, and :mod:`~schnetpack.generative.processes` owns it.
This module holds everything written on top of it in the language of
differential equations:

- :class:`SDE` — the second chart. When the endpoint is an independent
  isotropic Gaussian of declared scale, the same marginals are those of the
  linear forward SDE dx = f(t) x dt + g(t) dw, with f = d/dt log a and
  g^2 = -sigma^2 d/dt log SNR (moment matching against the Gaussian kernel
  p(x_t | x0) = N(a x0, sigma^2 I)), and the closed forms that are
  statements about that kernel — the perturbation kernel and the exact
  posterior that ancestral sampling and DDIM discretize.
- :class:`ReverseSDE` — the Anderson churn family reversing the chart:
  everything that crosses it, from the reverse-time SDE (churn = 1) to the
  probability-flow ODE (churn = 0).
- :class:`ReverseODE` — continuity-equation transport dx = v dt, which
  exists for *any* endpoint law and never touches the chart.

The chart is deliberately this module's *only* home: every chart object
exists exactly when the one-sided Gaussian kernel holds, so the check runs
once, at construction, instead of per method. A configuration without the
kernel — a shape prior, a value-dependent coupling, bridge noise — cannot
construct an :class:`SDE`, and the error names the obstruction
(:meth:`~schnetpack.generative.processes.Process.gaussian_kernel_obstruction`).
Acquire it through :meth:`~schnetpack.generative.processes.Process.sde`;
consumers that need it do so at their own construction, so an invalid
assembly fails there — not mid-run, and never silently. The routes that
never need the chart (velocity sampling at churn = 0, direct
x0/pseudo-force recovery via
:class:`~schnetpack.generative.sampler.DirectDenoisingSampler`) never call
it.

Reverse processes are never implemented per schedule; they split by
capability, and both take a single *bound field* — a score or a velocity,
callable (x, t) — rather than a (model, parametrization, cond) triple.
Composing that triple into the field, and picking the class from the
assembly (the ``(parametrization, churn)`` pair plus any integrator
demand), is the assembler's job —
:meth:`~schnetpack.generative.sampler.Sampler.denoise` for the head route,
or by hand for a ready field. Both reverse classes expose the
``drift``/``diffusion`` interface every integrator consumes; integration
runs backwards in time, so integrators pass dt < 0.
"""

from typing import Callable, Tuple

import torch

from schnetpack.generative.processes import Process, expand_t

__all__ = ["SDE", "ReverseSDE", "ReverseODE"]


class SDE:
    """
    The (f, g) chart of a process: the linear forward SDE whose conditional
    marginals match the interpolant's.

    Exists iff the one-sided Gaussian kernel p(x_t | x0) = N(a x0, sigma^2 I)
    holds for the process's configuration — construction *is* the check, and
    a failed construction names the obstruction
    (:meth:`~schnetpack.generative.processes.Process.gaussian_kernel_obstruction`).
    Stateless beyond the process reference: build it where it is consumed and
    hold it, or ask :meth:`Process.sde` again — both are cheap.
    """

    def __init__(self, process: Process):
        obstruction = process.gaussian_kernel_obstruction()
        if obstruction is not None:
            raise ValueError(
                f"No (f, g) SDE chart for this configuration: {obstruction}. "
                "The chart-free routes remain available: velocity sampling "
                "at churn = 0 and the direct x0/pseudo-force recovery "
                "(DirectDenoisingSampler)."
            )
        self.process = process

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient f(t) = a'/a of the forward SDE.

        Which is d/dt log a — see
        :meth:`~schnetpack.generative.processes.Process.log_a_dot`, where the
        quotient is avoided rather than computed.
        """
        return self.process.log_a_dot(t)

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
        knob.
        """
        process = self.process
        return -process.sigma(t) ** 2 * process.log_snr_dot(t)

    def kernel(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean coefficient and std of the perturbation kernel p(x_t | x0),
        i.e. (a(t), sigma(t)) with p(x_t | x0) = N(a x0, sigma^2 I).
        """
        return self.process.a(t), self.process.sigma(t)

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

    def x0_from_score(
        self, x_t: torch.Tensor, score: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Tweedie's formula on the kernel: x0 = (x_t + sigma^2 score) / a.

        A chart statement, like the kernel it inverts — which is why it lives
        here and not on a parametrization. Singular where a -> 0 (flow
        matching at t = 1, VP at large t). For a score obtained from an
        x0-predicting head the round trip is algebraically exact: the
        sigma^2 of Tweedie's two directions cancels.
        """
        a, sigma = self.kernel(t)
        return (x_t + expand_t(sigma**2, x_t) * score) / expand_t(a, x_t)


class ReverseSDE:
    """
    Time reversal of a forward SDE, as a one-parameter family

        dx = [ f x - 1/2 (1 + churn) g^2 score(x, t) ] dt + sqrt(churn) g dw,

    for churn in [0, 1]. Every member shares the forward marginals: churn = 1
    is the Anderson (1982) reverse-time SDE, churn = 0 the probability-flow
    ODE, and values in between trade discretization error against stochastic
    churn. It maps onto the usual eta knob by churn = eta^2, since the
    Anderson drift f x - 1/2 (1 + eta^2) g^2 s equals v - 1/2 eta^2 g^2 s.

    Pure chart math: the constructor takes the :class:`SDE` and a *bound
    score field* ``score_fn(x, t) -> score`` — no model, no parametrization,
    no cond. How a raw head output becomes the score is the assembler's
    business — :class:`~schnetpack.generative.sampler.Sampler` binds
    ``parametrization.to_score`` and the model into that callable; a ready
    score field passes straight in. Taking the :class:`SDE` states the chart
    dependency where it cannot be missed: no chart, no ReverseSDE. The one
    reverse route that never crosses the chart — a velocity head at
    churn = 0 — is :class:`ReverseODE`.

    Exposes the drift/diffusion interface every integrator consumes.
    Integration runs backwards in time, so integrators pass dt < 0.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable,
        churn: float = 1.0,
    ):
        """
        Args:
            sde: the (f, g) chart of the forward process being reversed —
                from :meth:`~schnetpack.generative.processes.Process.sde`
            score_fn: bound score field, callable (x, t) -> score of the
                marginal p_t; model, parametrization and conditioning are
                already composed inside
            churn: stochasticity in [0, 1]; 1 = reverse SDE, 0 = probability-flow
                ODE. Equals eta^2 of the Anderson family.
        """
        self.sde = sde
        self.process = sde.process
        self.score_fn = score_fn
        self.churn = churn

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion of the process being reversed, endpoint scale
        included. Integrators that need g^2 read it here.
        """
        return self.sde.g2(t)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Anderson drift f x - 1/2 (1 + churn) g^2 score, shaped like x.
        Costs one evaluation of the score field.
        """
        score = self.score_fn(x, t)
        f = expand_t(self.sde.f(t), x)
        g2 = expand_t(self.g2(t), x)
        return f * x - 0.5 * (1.0 + self.churn) * g2 * score

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion sqrt(churn) g(t), shaped like t; zero on the ODE."""
        if self.churn == 0.0:
            return torch.zeros_like(t)
        return torch.sqrt(self.churn * self.g2(t))

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Score of the marginal p_t — the bound field itself.

        Integrators normally need only :meth:`drift` and :meth:`diffusion`;
        this exists for the ones that discretize the reverse process through
        the chart's closed forms instead — the ancestral steps, which read
        the raw score directly or convert it through
        :meth:`SDE.x0_from_score`.
        """
        return self.score_fn(x, t)


class ReverseODE:
    """
    Deterministic reverse transport along the learned velocity: dx = v dt.

    The continuity equation makes this valid for any endpoint law — Gaussian
    or structured, chart or no chart (see docs_new/flow_matching_sde.md §8.2)
    — which is exactly why it must not depend on the :class:`SDE`. Like
    :class:`ReverseSDE` it takes a *bound field*: ``velocity_fn(x, t) -> v``,
    with model, parametrization and conditioning already composed inside.
    Guarding that the composition never crosses the chart
    (``velocity_needs_chart`` is False — a velocity head) is the assembler's
    job — :class:`~schnetpack.generative.sampler.Sampler` sends a score,
    noise or x0 head, which reaches the velocity through f and g^2, to
    :class:`ReverseSDE` even at churn = 0.
    """

    def __init__(self, velocity_fn: Callable):
        """
        Args:
            velocity_fn: bound velocity field, callable (x, t) -> velocity;
                must not cross the (f, g) chart
        """
        self.velocity_fn = velocity_fn

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The learned velocity, shaped like x. Costs one field evaluation."""
        return self.velocity_fn(x, t)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Zero — this is the ODE."""
        return torch.zeros_like(t)



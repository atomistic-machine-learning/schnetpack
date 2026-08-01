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
- :func:`reverse` — the assembly entry point that picks between the two.

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
capability, and :func:`reverse` dispatches from the assembly — the
``(parametrization, churn)`` pair plus any integrator demand — so callers
keep a single churn knob. Both reverse classes expose the
``drift``/``diffusion`` interface every integrator consumes; integration
runs backwards in time, so integrators pass dt < 0.
"""

from typing import Callable, Optional, Tuple

import torch

from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.processes import Process, expand_t

__all__ = ["SDE", "ReverseSDE", "ReverseODE", "reverse"]


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


class ReverseSDE:
    """
    Time reversal of a forward SDE, as a one-parameter family

        dx = [ v(x, t) - 1/2 churn g^2 score(x, t) ] dt + sqrt(churn) g dw,

    for churn in [0, 1]. Every member shares the forward marginals: churn = 1
    is the Anderson (1982) reverse-time SDE, churn = 0 the probability-flow
    ODE, and values in between trade discretization error against stochastic
    churn. It maps onto the usual eta knob by churn = eta^2, since
    v - 1/2 eta^2 g^2 s equals the Anderson drift f x - 1/2 (1 + eta^2) g^2 s.

    Everything here crosses the (f, g) chart — the churn > 0 members through
    g^2 outright, and even churn = 0 through the score-to-velocity conversion
    for non-velocity parametrizations. Taking the :class:`SDE` in the
    constructor states that dependency where it cannot be missed: no chart,
    no ReverseSDE. The one reverse route that never crosses the chart —
    a velocity head at churn = 0 — is :class:`ReverseODE`, and
    :func:`reverse` picks between them.

    Exposes the drift/diffusion interface every integrator consumes.
    Integration runs backwards in time, so integrators pass dt < 0.
    """

    def __init__(
        self,
        sde: SDE,
        parametrization: Parametrization,
        model: Callable,
        churn: float = 1.0,
        cond=None,
    ):
        """
        Args:
            sde: the (f, g) chart of the forward process being reversed —
                from :meth:`~schnetpack.generative.processes.Process.sde`
            parametrization: contract between the model output and the fields
            model: callable (x, t, cond) -> raw output in ``parametrization``
            churn: stochasticity in [0, 1]; 1 = reverse SDE, 0 = probability-flow
                ODE. Equals eta^2 of the Anderson family.
            cond: conditioning passed through to the model on every call
        """
        parametrization.validate(sde.process)
        self.sde = sde
        self.process = sde.process
        self.parametrization = parametrization
        self.model = model
        self.churn = churn
        self.cond = cond

    def g2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Squared diffusion of the process being reversed, endpoint scale
        included. Integrators that need g^2 read it here.
        """
        return self.sde.g2(t)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse drift, shaped like x. Costs one model evaluation."""
        raw = self.model(x, t, self.cond)
        velocity = self.parametrization.to_velocity(self.process, raw, x, t)
        if self.churn == 0.0:
            return velocity
        # Same raw output, second conversion — never a second model call.
        score = self.parametrization.to_score(self.process, raw, x, t)
        return velocity - 0.5 * self.churn * expand_t(self.g2(t), x) * score

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion sqrt(churn) g(t), shaped like t; zero on the ODE."""
        if self.churn == 0.0:
            return torch.zeros_like(t)
        return torch.sqrt(self.churn * self.g2(t))

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Score of the marginal p_t, with ``cond`` bound.

        Integrators normally need only :meth:`drift` and :meth:`diffusion`;
        this exists for the ones that discretize the reverse process in terms
        of the raw score, such as
        :class:`~schnetpack.generative.integrators.ancestral.AncestralDDPM`.
        """
        return self.parametrization.to_score(
            self.process, self.model(x, t, self.cond), x, t
        )

    def x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x0-estimate of the marginal at t, with ``cond`` bound.

        Like :meth:`score`, this exists for the integrators that discretize
        the reverse process through something other than drift/diffusion —
        here the exact Gaussian posterior p(x_s | x_t, x0), which
        :class:`~schnetpack.generative.integrators.ancestral.Ancestral`
        steps through with this estimate in x0's place, via :attr:`sde`.
        """
        return self.parametrization.to_x0(
            self.process, self.model(x, t, self.cond), x, t
        )


class ReverseODE:
    """
    Deterministic reverse transport along the learned velocity: dx = v dt.

    The continuity equation makes this valid for any endpoint law — Gaussian
    or structured, chart or no chart (see docs_new/flow_matching_sde.md §8.2)
    — which is exactly why it must not depend on the :class:`SDE`. The price
    of that generality: only parametrizations whose velocity conversion
    never crosses the chart qualify (``velocity_needs_chart`` is False — a
    velocity head). A score, noise or x0 head reaches the velocity through
    f and g^2, and belongs on :class:`ReverseSDE` even at churn = 0.
    """

    def __init__(
        self,
        process: Process,
        parametrization: Parametrization,
        model: Callable,
        cond=None,
    ):
        """
        Args:
            process: forward process being reversed; supplies the schedule
            parametrization: contract between the model output and the
                velocity; must have ``velocity_needs_chart = False``
            model: callable (x, t, cond) -> raw output in ``parametrization``
            cond: conditioning passed through to the model on every call
        """
        parametrization.validate(process)
        if parametrization.velocity_needs_chart:
            raise TypeError(
                f"{type(parametrization).__name__} reaches the velocity "
                "through the (f, g) chart (velocity_needs_chart is True), "
                "so its reverse process is a ReverseSDE — use "
                "reverse(process, parametrization, model, churn=0.0) to "
                "assemble the probability-flow ODE through the chart."
            )
        self.process = process
        self.parametrization = parametrization
        self.model = model
        self.cond = cond

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The learned velocity, shaped like x. Costs one model evaluation."""
        raw = self.model(x, t, self.cond)
        return self.parametrization.to_velocity(self.process, raw, x, t)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Zero — this is the ODE."""
        return torch.zeros_like(t)


def reverse(
    process: Process,
    parametrization: Parametrization,
    model: Callable,
    churn: float = 1.0,
    cond=None,
    require_sde: bool = False,
) -> "ReverseODE | ReverseSDE":
    """
    Assemble the reverse process for a ``(process, parametrization)`` pair.

    Returns the chart-free :class:`ReverseODE` exactly when nothing in the
    assembly needs the chart — churn = 0, a chart-free velocity head, and no
    integrator demand — and a :class:`ReverseSDE` otherwise, acquiring the
    chart via :meth:`~schnetpack.generative.processes.Process.sde` so that a
    configuration without the Gaussian kernel fails here, at assembly, with
    the obstruction named.

    Args:
        process: forward process the model was trained on
        parametrization: contract the model was trained under
        model: callable (x, t, cond) -> raw output in ``parametrization``
        churn: stochasticity in [0, 1]; 1 = reverse SDE, 0 = probability-flow
            ODE. Equals eta^2 of the Anderson family.
        cond: conditioning passed through to the model on every call
        require_sde: force the :class:`ReverseSDE` even at churn = 0 — for
            integrators that discretize through the chart's closed forms
            (ancestral steps) rather than drift/diffusion
    """
    if churn == 0.0 and not parametrization.velocity_needs_chart and not require_sde:
        return ReverseODE(process, parametrization, model, cond=cond)
    return ReverseSDE(process.sde(), parametrization, model, churn=churn, cond=cond)

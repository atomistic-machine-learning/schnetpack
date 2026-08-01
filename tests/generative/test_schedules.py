"""Schedule geometry of the Process subclasses, at the default unit scale."""

import copy
import math

import pytest
import torch

from schnetpack.generative import (
    FlowMatching,
    Process,
    VE,
    VELinear,
    VP,
    VPISSNR,
    expand_t,
)


def all_schedules():
    return [VP(), VE(), VELinear(), FlowMatching(), VPISSNR()]


def schedule_ids():
    return [type(p).__name__ for p in all_schedules()]


@pytest.fixture(params=all_schedules(), ids=schedule_ids())
def process(request):
    return request.param


@pytest.fixture
def vp():
    return VP()


@pytest.fixture
def ve():
    return VE()


def interior_times(process, n=5):
    """Times strictly inside [t_min, t_max], safe for finite differences."""
    lo, hi = process.t_min, process.t_max
    pad = 0.05 * (hi - lo)
    return torch.linspace(lo + pad, hi - pad, n, dtype=torch.float64)


# --- schedules and their derivatives ------------------------------------- #


def test_derivatives_match_finite_differences(process):
    t = interior_times(process)
    eps = 1e-6 * (process.t_max - process.t_min)

    d_a = (process.a(t + eps) - process.a(t - eps)) / (2 * eps)
    d_b = (process.b(t + eps) - process.b(t - eps)) / (2 * eps)

    assert torch.allclose(process.a_dot(t), d_a, rtol=1e-4, atol=1e-6)
    assert torch.allclose(process.b_dot(t), d_b, rtol=1e-4, atol=1e-6)


def test_g2_matches_db2_dt_minus_2_f_b2(process):
    # The general identity g^2 = d b^2/dt - 2 f b^2, by finite
    # differences. Reduces to g^2 = d b^2/dt only where f = 0. At the
    # default unit scale sigma = b, so the process g^2 is the identity's.
    t = interior_times(process)
    eps = 1e-6 * (process.t_max - process.t_min)

    db2 = (process.b(t + eps) ** 2 - process.b(t - eps) ** 2) / (2 * eps)
    expected = db2 - 2.0 * process.sde().f(t) * process.b(t) ** 2

    assert torch.allclose(process.sde().g2(t), expected, rtol=1e-4, atol=1e-6)


def test_vp_derived_drift_and_diffusion_match_analytic(vp):
    # The VP process is defined in the literature by f = -beta/2 and g^2 = beta.
    # Deriving both from (a, b) must reproduce exactly that.
    t = torch.linspace(0.05, 1.0, 10, dtype=torch.float64)
    assert torch.allclose(vp.sde().f(t), -0.5 * vp.beta(t), rtol=1e-6)
    assert torch.allclose(vp.sde().g2(t), vp.beta(t), rtol=1e-6)


def test_ve_derived_diffusion_matches_analytic(ve):
    t = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    expected = 2.0 * math.log(1.0 / ve.b_min) * ve.b(t) ** 2
    assert torch.allclose(ve.sde().g2(t), expected, rtol=1e-6)
    assert torch.allclose(ve.sde().f(t), torch.zeros_like(t), atol=1e-12)


def test_fm_diffusion_is_finite_on_usable_range():
    # g^2 = 2 t / (1 - t) blows up at t = 1; the default t_max keeps the
    # whole usable range finite.
    fm = FlowMatching()
    t = torch.linspace(fm.t_min, fm.t_max, 100, dtype=torch.float64)
    g2 = fm.sde().g2(t)
    assert torch.isfinite(g2).all()
    assert (g2 >= 0).all()

    expected = 2.0 * t / (1.0 - t)
    assert torch.allclose(g2, expected, rtol=1e-6)


# --- marginals ------------------------------------------------------------ #


def test_vp_marginals_variance_preserving(vp):
    t = torch.rand(1000) * vp.t_max
    a, b = vp.a_b(t)
    assert torch.allclose(a**2 + b**2, torch.ones_like(t), atol=1e-5)

    a0, s0 = vp.a_b(torch.tensor([0.0]))
    aT, sT = vp.a_b(torch.tensor([vp.t_max]))
    assert a0.item() == pytest.approx(1.0)
    assert s0.item() == pytest.approx(0.0, abs=1e-4)
    assert aT.item() < 1e-2
    assert sT.item() == pytest.approx(1.0, abs=1e-3)


def test_ve_marginals(ve):
    t = torch.tensor([0.0, ve.t_max])
    a, b = ve.a_b(t)
    assert torch.allclose(a, torch.ones_like(t))
    assert b[0].item() == pytest.approx(ve.b_min)
    assert b[1].item() == pytest.approx(1.0, rel=1e-6)


def test_b_is_normalized_to_one_at_t_max(process):
    # The convention the whole subpackage relies on: b(t_max) = 1 (within the
    # t_max approximation), so x_{t_max} is exactly the prior's endpoint and
    # the scale of the process lives on the prior alone.
    b_end = process.b(torch.tensor([process.t_max], dtype=torch.float64)).item()
    assert b_end == pytest.approx(1.0, abs=2e-3)

    t = torch.linspace(process.t_min, process.t_max, 50, dtype=torch.float64)
    assert (process.b(t) >= 0).all()
    assert (process.b(t) <= 1.0 + 1e-9).all()


def test_snr_and_log_snr_agree(process):
    t = interior_times(process)
    assert torch.allclose(process.snr(t), process.a(t) ** 2 / process.b(t) ** 2)
    assert torch.allclose(process.log_snr(t), torch.log(process.snr(t)), rtol=1e-5)


def test_interpolate_matches_marginal_stats(vp):
    # One-shot noising is interpolate on a drawn pair; for Gaussian x1 the
    # result must carry the closed-form marginal variance a^2 + b^2.
    torch.manual_seed(0)
    x0 = torch.randn(20000, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((20000,), 0.5)
    x_t = vp.interpolate(x0, x1, t)
    a, b = vp.a_b(t[:1])
    expected_std = math.sqrt(a.item() ** 2 + b.item() ** 2)
    assert x_t.std().item() == pytest.approx(expected_std, abs=0.02)


def test_per_sample_times_broadcast(vp):
    x0 = torch.randn(8, 5, 3)
    t = torch.rand(8) * 0.9 + 0.05

    x_t = vp.interpolate(x0, torch.randn_like(x0), t)
    assert x_t.shape == x0.shape
    assert vp.sde().g2(t).shape == t.shape
    assert vp.sde().f(t).shape == t.shape


# --- the two schedule routes ---------------------------------------------- #


def test_tv_and_snr_agree_with_a_b(process):
    # The TV/SNR pair, whichever way round the schedule defined itself.
    t = interior_times(process)
    a, b = process.a_b(t)
    assert torch.allclose(process.tv(t), a**2 + b**2, rtol=1e-6)
    assert torch.allclose(process.log_snr(t), torch.log(a**2 / b**2), rtol=1e-6)


def test_a_schedule_redefined_through_tv_snr_is_the_same_schedule(process):
    # The heart of it: read TV and log-SNR off a schedule defined by a/b,
    # feed them back through the other route, and the coefficients must return.
    # If the inversion were wrong this is what would catch it.
    class Mirror(Process):
        def tv(self, t):
            return process.tv(t)

        def log_snr(self, t):
            return process.log_snr(t)

    mirror = Mirror(t_min=process.t_min, t_max=process.t_max)
    t = interior_times(process)

    assert torch.allclose(mirror.a(t), process.a(t), rtol=1e-6, atol=1e-9)
    assert torch.allclose(mirror.b(t), process.b(t), rtol=1e-6, atol=1e-9)
    # and so must everything derived from them
    assert torch.allclose(mirror.sde().g2(t), process.sde().g2(t), rtol=1e-4, atol=1e-7)


def test_vp_issnr_is_variance_preserving():
    p = VPISSNR()
    t = interior_times(p, n=9)
    assert torch.allclose(p.tv(t), torch.ones_like(t))
    assert torch.allclose(p.a(t) ** 2 + p.b(t) ** 2, torch.ones_like(t))


def test_vp_issnr_recovers_flow_matching_a_b_up_to_the_tv_factor():
    # eta=2, kappa=0 gives log SNR = 2 log((1-t)/t), the SNR of the linear
    # interpolant. VP-ISSNR shares that SNR but flattens TV to 1, so its
    # coefficients are FlowMatching's divided by FlowMatching's total variance.
    p = VPISSNR(eta=2.0, kappa=0.0)
    fm = FlowMatching()
    t = interior_times(p, n=7)

    assert torch.allclose(p.log_snr(t), fm.log_snr(t), rtol=1e-6)

    tv = torch.sqrt(fm.tv(t))
    assert torch.allclose(p.a(t), fm.a(t) / tv, rtol=1e-6)
    assert torch.allclose(p.b(t), fm.b(t) / tv, rtol=1e-6)


def test_defining_neither_schedule_pair_is_a_type_error():
    # Both routes derive the other, so a process with neither would recurse.
    # That has to fail at class definition, naming the class, not at first call.
    with pytest.raises(TypeError, match="defines no schedule"):

        class Halfway(Process):
            def a(self, t):  # b missing -> neither pair is complete
                return torch.ones_like(t)


# --- the log-derivative identity ------------------------------------------ #


def test_log_derivatives_equal_the_quotients_they_replace(process):
    # d/dt log x = x_dot / x. Away from the endpoints both forms are fine and
    # must agree; the identity earns its keep where they are not.
    t = interior_times(process)
    assert torch.allclose(
        process.log_a_dot(t), process.a_dot(t) / process.a(t), rtol=1e-6
    )
    assert torch.allclose(
        process.log_b_dot(t), process.b_dot(t) / process.b(t), rtol=1e-6
    )
    assert torch.allclose(
        process.log_snr_dot(t),
        2.0 * (process.log_a_dot(t) - process.log_b_dot(t)),
        rtol=1e-6,
    )


def test_g2_equals_the_quotient_free_form(process):
    # g^2 = -sigma^2 d/dt log SNR is (at unit scale) the same number as
    # 2 b b' - 2 f b^2, which is what g2 no longer computes.
    t = interior_times(process)
    b = process.b(t)
    old_form = 2.0 * b * process.b_dot(t) - 2.0 * process.sde().f(t) * b**2
    assert torch.allclose(process.sde().g2(t), old_form, rtol=1e-6, atol=1e-9)


def test_log_snr_decreases_so_g2_is_non_negative(process):
    t = interior_times(process, n=9)
    assert (process.log_snr_dot(t) <= 0).all()
    assert (process.sde().g2(t) >= 0).all()


def test_a_closed_form_log_derivative_survives_a_0_over_0_quotient():
    # Why f routes through log_a_dot rather than dividing. Give a a
    # double zero and a_dot vanishes with it, so a_dot/a is 0/0 =
    # nan. A schedule that knows d/dt log a in closed form -- as VP knows
    # -beta/2 -- never forms the quotient and answers correctly.
    #
    # Note the autograd *default* would not save you here: it applies the chain
    # rule as (1/a) * a_dot, which is the same 0/0. The identity buys
    # the override, not magic.
    class DoubleZero(Process):
        def a(self, t):
            return (1.0 - t) ** 2

        def b(self, t):
            return t

        def log_a_dot(self, t):
            return -2.0 / (1.0 - t)  # closed form: a never appears

    p = DoubleZero(t_min=0.0, t_max=1.0)

    t = torch.tensor([0.5], dtype=torch.float64)
    assert torch.allclose(p.sde().f(t), torch.full_like(t, -4.0))  # agrees away from 0

    at_zero = torch.ones(1, dtype=torch.float64)  # a(1) == a_dot(1) == 0
    assert torch.isnan(p.a_dot(at_zero) / p.a(at_zero)).all()
    assert torch.isneginf(p.sde().f(at_zero)).all()


def test_g2_is_finite_where_the_old_form_needed_a_singular_f():
    # The TV/SNR payoff. VPISSNR(eta=4) has a ~ (1-t)^2 near t=1, so the
    # old g^2 = 2 b b' - 2 f b^2 needs f = a'/a -> 0/0 there.
    # Routing through log_snr_dot differentiates the schedule as written, so no
    # a is ever formed and g^2 stays finite.
    p = VPISSNR(eta=4.0)
    t = torch.tensor([1.0 - 1e-12], dtype=torch.float64)
    assert torch.isfinite(p.sde().g2(t)).all()
    assert (p.sde().g2(t) >= 0).all()


# --- autograd derivatives ------------------------------------------------- #


def test_autograd_derivatives_match_the_analytic_ones(process):
    # Every schedule here overrides a_dot/b_dot for speed. Strip the
    # overrides and the base class's autograd must reproduce them, which is what
    # makes those overrides an optimization rather than a second source of truth.
    Auto = type(
        f"Auto{type(process).__name__}",
        (type(process),),
        {"a_dot": Process.a_dot, "b_dot": Process.b_dot},
    )
    auto = copy.copy(process)
    auto.__class__ = Auto

    t = interior_times(process)
    assert torch.allclose(auto.a_dot(t), process.a_dot(t), rtol=1e-6, atol=1e-9)
    assert torch.allclose(auto.b_dot(t), process.b_dot(t), rtol=1e-6, atol=1e-9)


def test_autograd_derivatives_survive_no_grad(process):
    # Sampling runs under torch.no_grad(); the derivative still has to build its
    # own graph there or every reverse step would fail.
    t = interior_times(process)
    with torch.no_grad():
        assert torch.isfinite(Process.a_dot(process, t)).all()
        assert torch.isfinite(Process.b_dot(process, t)).all()


def test_autograd_derivative_of_a_constant_schedule_is_zero():
    # a = ones_like(t) has no grad_fn at all: autograd calls it unused, not
    # zero. VE-type schedules would crash on their own a without the fallback.
    ve = VE()
    t = interior_times(ve)
    assert torch.allclose(Process.a_dot(ve, t), torch.zeros_like(t))


def test_autograd_derivatives_return_a_plain_tensor(process):
    # No graph comes back out, even when the caller's t carries one. Diffuse
    # builds targets in the dataloader, and a target with a grad_fn cannot be
    # pickled to a worker process.
    t = interior_times(process).requires_grad_(True)
    for dot in (Process.a_dot(process, t), Process.b_dot(process, t)):
        assert not dot.requires_grad
        assert dot.grad_fn is None


# --- the gamma hook ------------------------------------------------------- #


def test_gamma_defaults_to_none_and_interpolate_is_exactly_two_term(process):
    assert process.gamma(torch.tensor([0.5])) is None

    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5 * (process.t_min + process.t_max))

    expected = expand_t(process.a(t), x0) * x0 + expand_t(process.b(t), x1) * x1
    assert torch.equal(process.interpolate(x0, x1, t), expected)


def test_two_term_interpolate_consumes_no_rng(process):
    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5 * (process.t_min + process.t_max))

    state = torch.random.get_rng_state()
    process.interpolate(x0, x1, t)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_bridge_schedule_gets_the_three_term_interpolant():
    class BridgeVP(VP):
        def gamma(self, t):
            return 0.5 * torch.ones_like(t)

    bridge = BridgeVP()
    x0, x1, eps = torch.randn(6, 3), torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5)

    expected = (
        expand_t(bridge.a(t), x0) * x0
        + expand_t(bridge.b(t), x1) * x1
        + 0.5 * eps
    )
    assert torch.allclose(bridge.interpolate(x0, x1, t, eps=eps), expected)


def test_bridge_schedule_draws_its_own_noise_when_not_given():
    class BridgeVP(VP):
        def gamma(self, t):
            return 0.5 * torch.ones_like(t)

    bridge = BridgeVP()
    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5)

    torch.manual_seed(0)
    a = bridge.interpolate(x0, x1, t)
    torch.manual_seed(1)
    b = bridge.interpolate(x0, x1, t)
    assert not torch.allclose(a, b)


# --- structure ------------------------------------------------------------ #


def test_schedules_expose_a_usable_time_range(process):
    assert process.t_min < process.t_max
    assert process.b(torch.tensor([process.t_min])).item() > 0.0


def test_process_is_abstract():
    with pytest.raises(TypeError):
        Process(t_min=0.0, t_max=1.0)


def test_the_parametrization_axis_does_not_leak_into_the_process(process):
    # The axis separation, pinned: a process knows its schedule, endpoint and
    # pairing, and nothing about what a network predicts. Targets, field
    # conversions and reverse processes belong to the parametrization. If one
    # of these comes back, an axis has leaked.
    for leaked in [
        "target_score",
        "target_eps",
        "target_x0",
        "target_velocity",
        "score_from_eps",
        "eps_from_score",
        "score_from_x0",
        "x0_from_score",
        "velocity_from_score",
        "score_from_velocity",
        "reverse",
        "probability_flow",
        "diffuse",
        "marginal_prob",
    ]:
        assert not hasattr(process, leaked), leaked

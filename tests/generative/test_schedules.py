"""Schedule geometry of the schedules, at the default unit scale."""

import copy
import math

import pytest
import torch

from schnetpack.generative import (
    SDE,
    VE,
    VP,
    FlowMatching,
    GaussianInterpolant,
    GaussianSchedule,
    Interpolant,
    Schedule,
    VELinear,
    expand_t,
)


def gaussian_schedules():
    return [VP(), VE(2e-4, 1.0), VELinear()]


def all_schedules():
    return gaussian_schedules() + [FlowMatching()]


def schedule_ids(schedules):
    return [type(s).__name__ for s in schedules]


@pytest.fixture(params=all_schedules(), ids=schedule_ids(all_schedules()))
def schedule(request):
    return request.param


@pytest.fixture(params=gaussian_schedules(), ids=schedule_ids(gaussian_schedules()))
def gaussian(request):
    return request.param


@pytest.fixture
def process(schedule):
    return Interpolant(schedule)


@pytest.fixture
def vp():
    return GaussianInterpolant(VP())


@pytest.fixture
def ve():
    return GaussianInterpolant(VE(2e-4, 1.0))


def interior_times(schedule, n=5):
    """Times strictly inside [t_min, t_max], safe for finite differences."""
    lo, hi = schedule.t_min, schedule.t_max
    pad = 0.05 * (hi - lo)
    return torch.linspace(lo + pad, hi - pad, n, dtype=torch.float64)


# --- schedules and their derivatives ------------------------------------- #


def test_derivatives_match_finite_differences(schedule):
    t = interior_times(schedule)
    eps = 1e-6 * (schedule.t_max - schedule.t_min)

    d_a = (schedule.a(t + eps) - schedule.a(t - eps)) / (2 * eps)
    d_b = (schedule.b(t + eps) - schedule.b(t - eps)) / (2 * eps)

    assert torch.allclose(schedule.a_dot(t), d_a, rtol=1e-4, atol=1e-6)
    assert torch.allclose(schedule.b_dot(t), d_b, rtol=1e-4, atol=1e-6)


def test_g2_matches_db2_dt_minus_2_f_b2(gaussian):
    # The general identity g^2 = d b^2/dt - 2 f b^2, by finite
    # differences. Reduces to g^2 = d b^2/dt only where f = 0. At the
    # default unit scale sigma = b, so the chart's g^2 is the identity's.
    t = interior_times(gaussian)
    eps = 1e-6 * (gaussian.t_max - gaussian.t_min)

    db2 = (gaussian.b(t + eps) ** 2 - gaussian.b(t - eps) ** 2) / (2 * eps)
    expected = db2 - 2.0 * SDE(gaussian).f(t) * gaussian.b(t) ** 2

    assert torch.allclose(SDE(gaussian).g2(t), expected, rtol=1e-4, atol=1e-6)


def test_vp_derived_drift_and_diffusion_match_analytic(vp):
    # The VP process is defined in the literature by f = -beta/2 and g^2 = beta.
    # Deriving both from (a, b) must reproduce exactly that.
    t = torch.linspace(0.05, 1.0, 10, dtype=torch.float64)
    beta = vp.schedule.beta(t)
    assert torch.allclose(SDE(vp.schedule).f(t), -0.5 * beta, rtol=1e-6)
    assert torch.allclose(SDE(vp.schedule).g2(t), beta, rtol=1e-6)


def test_ve_derived_diffusion_matches_analytic(ve):
    t = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    b_min = ve.schedule.sigma_min / ve.schedule.sigma_max
    expected = 2.0 * math.log(1.0 / b_min) * ve.schedule.b(t) ** 2
    assert torch.allclose(SDE(ve.schedule).g2(t), expected, rtol=1e-6)
    assert torch.allclose(SDE(ve.schedule).f(t), torch.zeros_like(t), atol=1e-12)


# --- marginals ------------------------------------------------------------ #


def test_vp_marginals_variance_preserving(vp):
    t = torch.rand(1000) * vp.schedule.t_max
    a, b = vp.schedule.a(t), vp.schedule.b(t)
    assert torch.allclose(a**2 + b**2, torch.ones_like(t), atol=1e-5)

    t0, tT = torch.tensor([0.0]), torch.tensor([vp.schedule.t_max])
    a0, s0 = vp.schedule.a(t0), vp.schedule.b(t0)
    aT, sT = vp.schedule.a(tT), vp.schedule.b(tT)
    assert a0.item() == pytest.approx(1.0)
    assert s0.item() == pytest.approx(0.0, abs=1e-4)
    assert aT.item() < 1e-2
    assert sT.item() == pytest.approx(1.0, abs=1e-3)


def test_ve_marginals(ve):
    t = torch.tensor([0.0, ve.schedule.t_max])
    a, b = ve.schedule.a(t), ve.schedule.b(t)
    assert torch.allclose(a, torch.ones_like(t))
    assert b[0].item() == pytest.approx(ve.schedule.sigma_min / ve.schedule.sigma_max)
    assert b[1].item() == pytest.approx(1.0, rel=1e-6)


def test_b_is_normalized_to_one_at_t_max(schedule):
    # The convention the whole subpackage relies on: b(t_max) = 1 (within the
    # t_max approximation), so x_{t_max} is exactly the prior's endpoint and
    # the scale of the interpolant lives on the prior alone.
    b_end = schedule.b(torch.tensor([schedule.t_max], dtype=torch.float64)).item()
    assert b_end == pytest.approx(1.0, abs=2e-3)

    t = torch.linspace(schedule.t_min, schedule.t_max, 50, dtype=torch.float64)
    assert (schedule.b(t) >= 0).all()
    assert (schedule.b(t) <= 1.0 + 1e-9).all()


def test_interpolate_matches_marginal_stats(vp):
    # One-shot noising is interpolate on a drawn pair; for Gaussian x1 the
    # result must carry the closed-form marginal variance a^2 + b^2.
    torch.manual_seed(0)
    x0 = torch.randn(20000, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((20000,), 0.5)
    x_t = vp.interpolate(x0, x1, t)
    a, b = vp.schedule.a(t[:1]), vp.schedule.b(t[:1])
    expected_std = math.sqrt(a.item() ** 2 + b.item() ** 2)
    assert x_t.std().item() == pytest.approx(expected_std, abs=0.02)


def test_per_sample_times_broadcast(vp):
    x0 = torch.randn(8, 5, 3)
    t = torch.rand(8) * 0.9 + 0.05

    x_t = vp.interpolate(x0, torch.randn_like(x0), t)
    assert x_t.shape == x0.shape
    assert SDE(vp.schedule).g2(t).shape == t.shape
    assert SDE(vp.schedule).f(t).shape == t.shape


# --- the log-derivative identity ------------------------------------------ #


def test_log_derivatives_equal_the_quotients_they_replace(schedule):
    # d/dt log x = x_dot / x. Away from the endpoints both forms are fine and
    # must agree; the identity earns its keep where they are not.
    t = interior_times(schedule)
    assert torch.allclose(
        schedule.log_a_dot(t), schedule.a_dot(t) / schedule.a(t), rtol=1e-6
    )


def test_g2_equals_the_quotient_free_form(gaussian):
    # g^2 = 2 sigma (sigma' - f sigma) is (at unit scale) the same number as
    # 2 b b' - 2 f b^2.
    t = interior_times(gaussian)
    b = gaussian.b(t)
    old_form = 2.0 * b * gaussian.b_dot(t) - 2.0 * SDE(gaussian).f(t) * b**2
    assert torch.allclose(SDE(gaussian).g2(t), old_form, rtol=1e-6, atol=1e-9)


def test_g2_is_non_negative(gaussian):
    t = interior_times(gaussian, n=9)
    assert (SDE(gaussian).g2(t) >= 0).all()


def test_a_closed_form_log_derivative_survives_a_0_over_0_quotient():
    # Why f routes through log_a_dot rather than dividing. Give a a
    # double zero and a_dot vanishes with it, so a_dot/a is 0/0 =
    # nan. A schedule that knows d/dt log a in closed form -- as VP knows
    # -beta/2 -- never forms the quotient and answers correctly.
    #
    # Note the autograd *default* would not save you here: it applies the chain
    # rule as (1/a) * a_dot, which is the same 0/0. The identity buys
    # the override, not magic.
    class DoubleZero(GaussianSchedule):
        def a(self, t):
            return (1.0 - t) ** 2

        def sigma(self, t):
            return t

        def log_a_dot(self, t):
            return -2.0 / (1.0 - t)  # closed form: a never appears

    p = DoubleZero(sigma_max=1.0, t_min=0.0, t_max=1.0)

    t = torch.tensor([0.5], dtype=torch.float64)
    # agrees away from 0
    assert torch.allclose(SDE(p).f(t), torch.full_like(t, -4.0))

    at_zero = torch.ones(1, dtype=torch.float64)  # a(1) == a_dot(1) == 0
    assert torch.isnan(p.a_dot(at_zero) / p.a(at_zero)).all()
    assert torch.isneginf(SDE(p).f(at_zero)).all()


# --- autograd derivatives ------------------------------------------------- #


def test_autograd_derivatives_match_the_analytic_ones(schedule):
    # Every schedule here overrides a_dot/b_dot for speed. Strip the
    # overrides and the base class's autograd must reproduce them, which is what
    # makes those overrides an optimization rather than a second source of truth.
    Auto = type(
        f"Auto{type(schedule).__name__}",
        (type(schedule),),
        {"a_dot": Schedule.a_dot, "b_dot": Schedule.b_dot},
    )
    auto = copy.copy(schedule)
    auto.__class__ = Auto

    t = interior_times(schedule)
    assert torch.allclose(auto.a_dot(t), schedule.a_dot(t), rtol=1e-6, atol=1e-9)
    assert torch.allclose(auto.b_dot(t), schedule.b_dot(t), rtol=1e-6, atol=1e-9)


def test_autograd_derivatives_survive_no_grad(schedule):
    # Sampling runs under torch.no_grad(); the derivative still has to build its
    # own graph there or every reverse step would fail.
    t = interior_times(schedule)
    with torch.no_grad():
        assert torch.isfinite(Schedule.a_dot(schedule, t)).all()
        assert torch.isfinite(Schedule.b_dot(schedule, t)).all()


def test_autograd_derivative_of_a_constant_schedule_is_zero():
    # a = ones_like(t) has no grad_fn at all: autograd calls it unused, not
    # zero. VE-type schedules would crash on their own a without the fallback.
    ve = VE(2e-4, 1.0)
    t = interior_times(ve)
    assert torch.allclose(Schedule.a_dot(ve, t), torch.zeros_like(t))


def test_autograd_derivatives_return_a_plain_tensor(schedule):
    # No graph comes back out, even when the caller's t carries one. Diffuse
    # builds targets in the dataloader, and a target with a grad_fn cannot be
    # pickled to a worker process.
    t = interior_times(schedule).requires_grad_(True)
    for dot in (Schedule.a_dot(schedule, t), Schedule.b_dot(schedule, t)):
        assert not dot.requires_grad
        assert dot.grad_fn is None


# --- the interpolant ------------------------------------------------------ #


def test_interpolate_is_exactly_two_term(process):
    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5 * (process.schedule.t_min + process.schedule.t_max))

    expected = (
        expand_t(process.schedule.a(t), x0) * x0
        + expand_t(process.schedule.b(t), x1) * x1
    )
    assert torch.equal(process.interpolate(x0, x1, t), expected)


def test_two_term_interpolate_consumes_no_rng(process):
    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5 * (process.schedule.t_min + process.schedule.t_max))

    state = torch.random.get_rng_state()
    process.interpolate(x0, x1, t)
    assert torch.equal(torch.random.get_rng_state(), state)


# --- structure ------------------------------------------------------------ #


def test_schedules_expose_a_usable_time_range(schedule):
    assert schedule.t_min < schedule.t_max
    assert schedule.b(torch.tensor([schedule.t_min])).item() > 0.0


def test_schedule_is_abstract():
    with pytest.raises(TypeError):
        Schedule(t_min=0.0, t_max=1.0)


def test_the_parametrization_axis_does_not_leak_into_the_process(process):
    # The axis separation, pinned: an interpolant knows its schedule, endpoint and
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

import math

import pytest
import torch

from schnetpack.generative import (
    EDMPath,
    FMPath,
    Path,
    VELinearPath,
    VEPath,
    VPPath,
    expand_t,
)


def all_paths():
    return [VPPath(), VEPath(), VELinearPath(), FMPath(), EDMPath()]


def path_ids():
    return [type(p).__name__ for p in all_paths()]


@pytest.fixture(params=all_paths(), ids=path_ids())
def path(request):
    return request.param


@pytest.fixture
def vp():
    return VPPath()


@pytest.fixture
def ve():
    return VEPath()


def interior_times(path, n=5):
    """Times strictly inside [t_min, t_max], safe for finite differences."""
    lo, hi = path.t_min, path.t_max
    pad = 0.05 * (hi - lo)
    return torch.linspace(lo + pad, hi - pad, n, dtype=torch.float64)


# --- schedules and their derivatives ------------------------------------- #


def test_derivatives_match_finite_differences(path):
    t = interior_times(path)
    eps = 1e-6 * (path.t_max - path.t_min)

    d_alpha = (path.alpha(t + eps) - path.alpha(t - eps)) / (2 * eps)
    d_sigma = (path.sigma(t + eps) - path.sigma(t - eps)) / (2 * eps)

    assert torch.allclose(path.alpha_dot(t), d_alpha, rtol=1e-4, atol=1e-6)
    assert torch.allclose(path.sigma_dot(t), d_sigma, rtol=1e-4, atol=1e-6)


def test_g2_matches_dsigma2_dt_minus_2_f_sigma2(path):
    # The general identity g^2 = d sigma^2/dt - 2 f sigma^2, by finite
    # differences. Reduces to g^2 = d sigma^2/dt only where f = 0.
    t = interior_times(path)
    eps = 1e-6 * (path.t_max - path.t_min)

    dsigma2 = (path.sigma(t + eps) ** 2 - path.sigma(t - eps) ** 2) / (2 * eps)
    expected = dsigma2 - 2.0 * path.f(t) * path.sigma(t) ** 2

    assert torch.allclose(path.g2(t), expected, rtol=1e-4, atol=1e-6)


def test_vp_derived_drift_and_diffusion_match_analytic(vp):
    # The VP process is defined in the literature by f = -beta/2 and g^2 = beta.
    # Deriving both from (alpha, sigma) must reproduce exactly that.
    t = torch.linspace(0.05, 1.0, 10, dtype=torch.float64)
    assert torch.allclose(vp.f(t), -0.5 * vp.beta(t), rtol=1e-6)
    assert torch.allclose(vp.g2(t), vp.beta(t), rtol=1e-6)


def test_ve_derived_diffusion_matches_analytic(ve):
    t = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    expected = 2.0 * math.log(ve.sigma_max / ve.sigma_min) * ve.sigma(t) ** 2
    assert torch.allclose(ve.g2(t), expected, rtol=1e-6)
    assert torch.allclose(ve.f(t), torch.zeros_like(t), atol=1e-12)


def test_edm_drift_and_diffusion():
    edm = EDMPath()
    t = torch.linspace(0.1, 80.0, 10, dtype=torch.float64)
    assert torch.allclose(edm.f(t), torch.zeros_like(t), atol=1e-12)
    assert torch.allclose(edm.g2(t), 2.0 * t, rtol=1e-6)


def test_fm_diffusion_is_finite_on_usable_range():
    # g^2 = 2 t sigma_max^2 / (1 - t) blows up at t = 1; the default t_max
    # keeps the whole usable range finite.
    fm = FMPath()
    t = torch.linspace(fm.t_min, fm.t_max, 100, dtype=torch.float64)
    g2 = fm.g2(t)
    assert torch.isfinite(g2).all()
    assert (g2 >= 0).all()

    expected = 2.0 * t * fm.sigma_max**2 / (1.0 - t)
    assert torch.allclose(g2, expected, rtol=1e-6)


# --- marginals ------------------------------------------------------------ #


def test_vp_marginals_variance_preserving(vp):
    t = torch.rand(1000) * vp.t_max
    alpha, sigma = vp.alpha_sigma(t)
    assert torch.allclose(alpha**2 + sigma**2, torch.ones_like(t), atol=1e-5)

    a0, s0 = vp.alpha_sigma(torch.tensor([0.0]))
    aT, sT = vp.alpha_sigma(torch.tensor([vp.t_max]))
    assert a0.item() == pytest.approx(1.0)
    assert s0.item() == pytest.approx(0.0, abs=1e-4)
    assert aT.item() < 1e-2
    assert sT.item() == pytest.approx(1.0, abs=1e-3)


def test_ve_marginals(ve):
    t = torch.tensor([0.0, ve.t_max])
    alpha, sigma = ve.alpha_sigma(t)
    assert torch.allclose(alpha, torch.ones_like(t))
    assert sigma[0].item() == pytest.approx(ve.sigma_min)
    assert sigma[1].item() == pytest.approx(ve.sigma_max, rel=1e-4)


def test_snr_and_log_snr_agree(path):
    t = interior_times(path)
    assert torch.allclose(path.snr(t), path.alpha(t) ** 2 / path.sigma(t) ** 2)
    assert torch.allclose(path.log_snr(t), torch.log(path.snr(t)), rtol=1e-5)


def test_diffuse_matches_marginal_stats(vp):
    torch.manual_seed(0)
    x0 = torch.randn(20000, 1)
    t = torch.full((20000,), 0.5)
    x_t, noise = vp.diffuse(x0, t)
    alpha, sigma = vp.alpha_sigma(t[:1])
    expected_std = math.sqrt(alpha.item() ** 2 + sigma.item() ** 2)
    assert x_t.std().item() == pytest.approx(expected_std, abs=0.02)
    assert noise.shape == x0.shape


def test_per_sample_times_broadcast(vp):
    x0 = torch.randn(8, 5, 3)
    t = torch.rand(8) * 0.9 + 0.05

    x_t, _ = vp.diffuse(x0, t)
    assert x_t.shape == x0.shape
    assert vp.g2(t).shape == t.shape
    assert vp.f(t).shape == t.shape


# --- the gamma hook ------------------------------------------------------- #


def test_gamma_defaults_to_none_and_interpolate_is_exactly_two_term(path):
    assert path.gamma(torch.tensor([0.5])) is None

    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5 * (path.t_min + path.t_max))

    expected = expand_t(path.alpha(t), x0) * x0 + expand_t(path.sigma(t), x1) * x1
    assert torch.equal(path.interpolate(x0, x1, t), expected)


def test_two_term_interpolate_consumes_no_rng(path):
    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5 * (path.t_min + path.t_max))

    state = torch.random.get_rng_state()
    path.interpolate(x0, x1, t)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_bridge_path_gets_the_three_term_interpolant():
    class BridgePath(VPPath):
        def gamma(self, t):
            return 0.5 * torch.ones_like(t)

    bridge = BridgePath()
    x0, x1, eps = torch.randn(6, 3), torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5)

    expected = (
        expand_t(bridge.alpha(t), x0) * x0
        + expand_t(bridge.sigma(t), x1) * x1
        + 0.5 * eps
    )
    assert torch.allclose(bridge.interpolate(x0, x1, t, eps=eps), expected)


def test_bridge_path_draws_its_own_noise_when_not_given():
    class BridgePath(VPPath):
        def gamma(self, t):
            return 0.5 * torch.ones_like(t)

    bridge = BridgePath()
    x0, x1 = torch.randn(6, 3), torch.randn(6, 3)
    t = torch.full((6,), 0.5)

    torch.manual_seed(0)
    a = bridge.interpolate(x0, x1, t)
    torch.manual_seed(1)
    b = bridge.interpolate(x0, x1, t)
    assert not torch.allclose(a, b)


# --- structure ------------------------------------------------------------ #


def test_paths_expose_a_usable_time_range(path):
    assert path.t_min < path.t_max
    assert path.sigma(torch.tensor([path.t_min])).item() > 0.0


def test_path_is_abstract():
    with pytest.raises(TypeError):
        Path(t_min=0.0, t_max=1.0)


def test_path_is_only_geometry(path):
    # The axis separation, pinned: a path knows its schedule and what derives
    # from it, and nothing about what a network predicts. Targets, field
    # conversions and reverse processes belong to the parametrization; priors
    # to the prior classes. If one of these comes back, an axis has leaked.
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
        "sample_prior",
    ]:
        assert not hasattr(path, leaked), leaked

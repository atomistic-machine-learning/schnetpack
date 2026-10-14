import pytest
import torch

from schnetpack.generative import (
    SDE,
    VE,
    VP,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    IdentityCoupling,
    Interpolant,
    PCVarianceCoupling,
    PermutationCoupling,
    VELinear,
)

# --- the forward move: prior -> coupling -> schedule ----------------------- #


def test_perturb_routes_prior_then_coupling_then_schedule():
    # The three moves, in order: the prior draws x1, the coupling pairs it,
    # the schedule places it. A recording coupling proves the pairing ran on
    # the prior's draw.
    seen = {}

    class RecordingCoupling(IdentityCoupling):
        def pair(self, x0, x1, groups=None):
            seen["x1"] = x1
            return x0, x1

    torch.manual_seed(0)
    process = Interpolant(
        VE(6e-3, 30.0), prior=GaussianPrior(std=30.0), coupling=RecordingCoupling()
    )
    x0 = torch.randn(64, 3)
    x_t, out0, x1, t = process.perturb(x0)

    assert torch.equal(out0, x0)
    assert seen["x1"].std().item() == pytest.approx(30.0, rel=0.1)  # the prior's scale
    # x_t is a x0 + b x1 with a = 1 on VE
    assert torch.allclose(x_t, x0 + process.schedule.b(t).reshape(-1, 1) * x1)


def test_perturb_accepts_given_times():
    process = GaussianInterpolant(VP())
    t = torch.full((8,), 0.3)
    _, _, _, t_out = process.perturb(torch.randn(8, 3), t=t)
    assert torch.equal(t_out, t)


def test_sample_t_stays_inside_the_usable_range():
    process = GaussianInterpolant(VP())
    t = process.sample_t(8192)
    assert t.min().item() >= process.schedule.t_min
    assert t.max().item() <= process.schedule.t_max
    assert t.min().item() < process.schedule.t_min + 0.05
    assert t.max().item() > process.schedule.t_max - 0.05


# --- scale lives on the prior, exposed by the interpolant ----------------- #


def test_sigma_is_b_times_the_prior_std():
    process = GaussianInterpolant(VE(6e-3, 30.0))
    t = torch.linspace(0.1, 0.9, 5)
    assert torch.allclose(process.schedule.sigma(t), process.schedule.b(t) * 30.0)


def test_ve_sigma_route_reproduces_the_classic_schedule():
    # VE(sigma_min, sigma_max) does the dimensionless split once: the
    # resulting sigma(t) is exactly sigma_min^(1-t) sigma_max^t.
    smin, smax = 0.3, 30.0
    process = GaussianInterpolant(VE(smin, smax))
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    expected = smin ** (1.0 - t) * smax**t
    assert torch.allclose(process.schedule.sigma(t), expected, rtol=1e-6)


def test_g2_scales_with_std_squared():
    unit = GaussianInterpolant(VE(2e-4, 1.0))
    scaled = GaussianInterpolant(VE(7.0 * 2e-4, 7.0))
    t = torch.linspace(0.1, 0.9, 5)
    assert torch.allclose(
        SDE(scaled.schedule).g2(t), 49.0 * SDE(unit.schedule).g2(t), rtol=1e-6
    )


def test_ve_builds_its_gaussian_endpoint_from_sigma_max():
    process = GaussianInterpolant(VE(0.3, 30.0))
    assert isinstance(process.prior, GaussianPrior)
    assert process.prior.std == 30.0
    assert process.schedule.b(torch.tensor(0.0)).item() == pytest.approx(0.01)


def test_vp_has_a_unit_gaussian_endpoint():
    process = GaussianInterpolant(VP())
    assert isinstance(process.prior, GaussianPrior)
    assert process.prior.std == 1.0


def test_gaussian_b_is_sigma_over_sigma_max():
    schedule = GaussianInterpolant(VELinear(sigma_max=4.0)).schedule
    t = torch.linspace(0.1, 0.9, 5)
    assert schedule.sigma_max == 4.0
    assert torch.allclose(schedule.b(t), schedule.sigma(t) / schedule.sigma_max)


def test_gaussian_interpolant_refuses_a_prior_std_other_than_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        GaussianInterpolant(VE(0.3, 30.0), prior=GaussianPrior(std=1.0))


def test_gaussian_interpolant_refuses_a_non_gaussian_prior():
    class NonGaussian(GaussianPrior):
        gaussian = False

    with pytest.raises(ValueError, match="isotropic Gaussian"):
        GaussianInterpolant(VELinear(), prior=NonGaussian(1.0))


def test_gaussian_interpolant_refuses_a_plain_schedule():
    # FlowMatching defines no sigma(t); it belongs in a plain Interpolant.
    with pytest.raises(TypeError, match="GaussianSchedule"):
        GaussianInterpolant(FlowMatching(), prior=GaussianPrior(std=3.0))


# --- the sampling start derives from the interpolant ----------------------- #


def test_sampling_prior_is_the_training_prior_when_the_coupling_preserves_it():
    prior = GaussianPrior(50.0)
    process = Interpolant(FlowMatching(), prior=prior, coupling=PermutationCoupling())
    assert process.sampling_prior() is prior


def test_sampling_prior_refuses_a_marginal_changing_coupling():
    process = Interpolant(VE(2e-4, 1.0), coupling=PCVarianceCoupling())
    with pytest.raises(ValueError, match="marginal"):
        process.sampling_prior()


# --- Gaussian-only closed forms, on the chart ------------------------------ #


def test_the_chart_refuses_a_plain_schedule():
    # The closed forms live on the chart, and a schedule without sigma(t)
    # cannot construct it — one refusal, at acquisition, instead of a check
    # per closed form.
    process = Interpolant(FlowMatching())
    with pytest.raises(TypeError, match="chart"):
        SDE(process.schedule)


def test_posterior_matches_the_ve_closed_form():
    # For VE (a = 1) the posterior mean is the convex blend
    # (sigma_s^2/sigma_t^2) x_t + (1 - sigma_s^2/sigma_t^2) x0, and its
    # variance is sigma_s^2 (sigma_t^2 - sigma_s^2)/sigma_t^2.
    process = GaussianInterpolant(VE(0.3, 30.0))
    t = torch.full((16,), 0.6, dtype=torch.float64)
    s = torch.full((16,), 0.4, dtype=torch.float64)
    x_t = torch.randn(16, 1, dtype=torch.float64)
    x0 = torch.randn(16, 1, dtype=torch.float64)

    mean, std = SDE(process.schedule).posterior(x_t, x0, t, s)

    sig_t = process.schedule.sigma(t).reshape(-1, 1)
    sig_s = process.schedule.sigma(s).reshape(-1, 1)
    w = sig_s**2 / sig_t**2
    expected_mean = w * x_t + (1.0 - w) * x0
    expected_var = (sig_s**2 * (sig_t**2 - sig_s**2) / sig_t**2).reshape(-1)

    assert torch.allclose(mean, expected_mean, rtol=1e-8)
    assert torch.allclose(std**2, expected_var, rtol=1e-8)


def test_posterior_recovers_the_ddpm_marginal_variance_on_vp():
    # Sanity on VP: the posterior variance is non-negative and the mean is a
    # genuine interpolation (bounded by the two inputs is not required, but
    # finiteness and the s->0 collapse are).
    process = GaussianInterpolant(VP())
    t = torch.full((8,), 0.5, dtype=torch.float64)
    s = torch.full((8,), 0.5 - 1e-6, dtype=torch.float64)
    x_t = torch.randn(8, 1, dtype=torch.float64)
    x0 = torch.randn(8, 1, dtype=torch.float64)
    mean, std = SDE(process.schedule).posterior(x_t, x0, t, s)
    # as s -> t the posterior concentrates on x_t
    assert torch.allclose(mean, x_t, atol=1e-4)
    assert std.max().item() < 1e-2

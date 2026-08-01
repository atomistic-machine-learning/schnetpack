import pytest
import torch

from schnetpack.generative import (
    FlowMatching,
    GaussianPrior,
    IdentityCoupling,
    PCVarianceCoupling,
    PermutationCoupling,
    VE,
    VP,
)


# --- the forward move: prior -> coupling -> schedule ----------------------- #


def test_perturb_routes_prior_then_coupling_then_schedule():
    # The three moves, in order: the prior draws x1, the coupling pairs it,
    # the schedule places it. A recording coupling proves the pairing ran on
    # the prior's draw.
    seen = {}

    class RecordingCoupling(IdentityCoupling):
        def pair(self, x0, x1):
            seen["x1"] = x1
            return x0, x1

    torch.manual_seed(0)
    process = VE(prior=GaussianPrior(30.0), coupling=RecordingCoupling())
    x0 = torch.randn(64, 3)
    x_t, out0, x1, t, eps = process.perturb(x0)

    assert torch.equal(out0, x0)
    assert seen["x1"].std().item() == pytest.approx(30.0, rel=0.1)  # the prior's scale
    assert eps is None  # no bridge noise on a VE schedule
    # x_t is a x0 + b x1 with a = 1 on VE
    assert torch.allclose(x_t, x0 + process.b(t).reshape(-1, 1) * x1)


def test_perturb_returns_the_bridge_noise_it_drew():
    # The reason the process owns the draw: a bridge target needs the exact
    # eps that entered the interpolant, so perturb must hand it back.
    class BridgeVE(VE):
        def gamma(self, t):
            return 0.1 * torch.ones_like(t)

    torch.manual_seed(0)
    process = BridgeVE()
    x0 = torch.randn(32, 3)
    x_t, _, x1, t, eps = process.perturb(x0)

    assert eps is not None and eps.shape == x0.shape
    # x_t must be reproducible from exactly that eps
    rebuilt = process.interpolate(x0, x1, t, eps=eps)
    assert torch.allclose(x_t, rebuilt)


def test_perturb_accepts_given_endpoints_but_still_pairs_them():
    torch.manual_seed(0)
    process = VE(coupling=PermutationCoupling())
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    _, _, paired, _, _ = process.perturb(x0, x1=x1)
    # same multiset, possibly reordered
    assert torch.allclose(
        x1[x1[:, 0].argsort()], paired[paired[:, 0].argsort()]
    )


def test_perturb_accepts_given_times():
    process = VP()
    t = torch.full((8,), 0.3)
    _, _, _, t_out, _ = process.perturb(torch.randn(8, 3), t=t)
    assert torch.equal(t_out, t)


def test_sample_t_stays_inside_the_usable_range():
    process = VP()
    t = process.sample_t(8192)
    assert t.min().item() >= process.t_min
    assert t.max().item() <= process.t_max
    assert t.min().item() < process.t_min + 0.05
    assert t.max().item() > process.t_max - 0.05


# --- scale lives on the prior, exposed by the process --------------------- #


def test_sigma_is_b_times_the_prior_std():
    process = VE(scale=30.0)
    t = torch.linspace(0.1, 0.9, 5)
    assert torch.allclose(process.sigma(t), process.b(t) * 30.0)


def test_ve_sigma_route_reproduces_the_classic_schedule():
    # VE(sigma_min, sigma_max) does the dimensionless split once: the
    # resulting sigma(t) is exactly sigma_min^(1-t) sigma_max^t.
    smin, smax = 0.3, 30.0
    process = VE(smin, smax)
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    expected = smin ** (1.0 - t) * smax**t
    assert torch.allclose(process.sigma(t), expected, rtol=1e-6)


def test_g2_scales_with_std_squared():
    unit = VE()
    scaled = VE(scale=7.0)
    t = torch.linspace(0.1, 0.9, 5)
    assert torch.allclose(scaled.sde().g2(t), 49.0 * unit.sde().g2(t), rtol=1e-6)


def test_sigma_and_the_chart_raise_without_a_declared_scale():
    # A shape-prior-like endpoint (std=None) has no single noise level; sigma
    # must say so rather than return garbage, and the (f, g) chart — whose
    # g^2 needs sigma — must refuse to exist at all.
    class NoScalePrior(GaussianPrior):
        def __init__(self):
            self.std = None
            self.gaussian = True

    process = VE(prior=NoScalePrior())
    t = torch.linspace(0.1, 0.9, 5)
    with pytest.raises(ValueError, match="no scalar endpoint scale"):
        process.sigma(t)
    with pytest.raises(ValueError, match="no scalar endpoint scale"):
        process.sde()


def test_scale_and_prior_together_are_refused():
    with pytest.raises(TypeError, match="not both"):
        VP(scale=2.0, prior=GaussianPrior(2.0))


def test_ve_sigma_pair_excludes_the_dimensionless_route():
    with pytest.raises(TypeError, match="not both"):
        VE(0.3, 30.0, b_min=1e-2)
    with pytest.raises(TypeError, match="not both"):
        VE(0.3, 30.0, prior=GaussianPrior(30.0))
    with pytest.raises(TypeError, match="together"):
        VE(sigma_min=0.3)


# --- the sampling start derives from the process -------------------------- #


def test_sampling_prior_is_the_training_prior_when_the_coupling_preserves_it():
    prior = GaussianPrior(50.0)
    process = VE(prior=prior, coupling=PermutationCoupling())
    assert process.sampling_prior() is prior


def test_sampling_prior_refuses_a_marginal_changing_coupling():
    process = VE(coupling=PCVarianceCoupling())
    with pytest.raises(ValueError, match="marginal"):
        process.sampling_prior()


# --- the Gaussian kernel is judged from the configuration ------------------ #


def test_default_assemblies_have_the_gaussian_kernel():
    for process in [VP(), VE(0.3, 30.0), FlowMatching()]:
        assert process.has_gaussian_kernel
        assert process.gaussian_kernel_obstruction() is None


def test_only_value_independent_couplings_keep_the_gaussian_kernel():
    # The kernel is a statement about the conditional p(x1 | x0), so marginal
    # preservation is not enough: an optimal assignment permutes exchangeable
    # draws (marginal survives) but hands each x0 its closest one
    # (conditional does not). The configuration decides, not the class.
    assert VP(coupling=IdentityCoupling()).has_gaussian_kernel
    repaired = VP(coupling=PermutationCoupling())
    assert not repaired.has_gaussian_kernel
    assert "depending on the values" in repaired.gaussian_kernel_obstruction()
    # ... while the sampling start still derives, because the marginal holds
    assert repaired.sampling_prior() is repaired.prior


def test_a_non_gaussian_prior_obstructs_the_kernel():
    class NonGaussian(GaussianPrior):
        gaussian = False

    process = VE(prior=NonGaussian(1.0))
    assert not process.has_gaussian_kernel
    assert "isotropic Gaussian" in process.gaussian_kernel_obstruction()


def test_a_prior_without_a_scale_obstructs_the_kernel():
    class NoScale(GaussianPrior):
        def __init__(self):
            self.std = None

    process = VE(prior=NoScale())
    assert not process.has_gaussian_kernel
    assert "scalar endpoint scale" in process.gaussian_kernel_obstruction()


def test_a_marginal_changing_coupling_obstructs_the_kernel():
    process = VE(scale=30.0, coupling=PCVarianceCoupling())
    assert not process.has_gaussian_kernel
    assert "depending on the values" in process.gaussian_kernel_obstruction()


def test_bridge_noise_obstructs_the_kernel():
    class BridgeVP(VP):
        def gamma(self, t):
            return 0.1 * torch.ones_like(t)

    process = BridgeVP()
    assert not process.has_gaussian_kernel
    assert "bridge noise" in process.gaussian_kernel_obstruction()


def test_zero_bridge_noise_does_not_obstruct():
    # A gamma that returns zeros carries no latent — the probe must check the
    # values, not just non-None.
    class ZeroBridgeVP(VP):
        def gamma(self, t):
            return torch.zeros_like(t)

    assert ZeroBridgeVP().has_gaussian_kernel


# --- Gaussian-only closed forms, on the chart ------------------------------ #


def test_kernel_matches_the_perturbation_kernel():
    process = VE(0.3, 30.0)
    t = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)
    a, sigma = process.sde().kernel(t)
    assert torch.allclose(a, process.a(t))
    assert torch.allclose(sigma, process.sigma(t))


def test_the_chart_refuses_without_the_gaussian_kernel():
    # The closed forms live on the chart, and a configuration without the
    # kernel cannot construct it — one refusal, at acquisition, instead of a
    # check per closed form.
    process = VE(scale=30.0, coupling=PCVarianceCoupling())
    with pytest.raises(ValueError, match="chart"):
        process.sde()


def test_posterior_matches_the_ve_closed_form():
    # For VE (a = 1) the posterior mean is the convex blend
    # (sigma_s^2/sigma_t^2) x_t + (1 - sigma_s^2/sigma_t^2) x0, and its
    # variance is sigma_s^2 (sigma_t^2 - sigma_s^2)/sigma_t^2.
    process = VE(0.3, 30.0)
    t = torch.full((16,), 0.6, dtype=torch.float64)
    s = torch.full((16,), 0.4, dtype=torch.float64)
    x_t = torch.randn(16, 1, dtype=torch.float64)
    x0 = torch.randn(16, 1, dtype=torch.float64)

    mean, std = process.sde().posterior(x_t, x0, t, s)

    sig_t = process.sigma(t).reshape(-1, 1)
    sig_s = process.sigma(s).reshape(-1, 1)
    w = sig_s**2 / sig_t**2
    expected_mean = w * x_t + (1.0 - w) * x0
    expected_var = (sig_s**2 * (sig_t**2 - sig_s**2) / sig_t**2).reshape(-1)

    assert torch.allclose(mean, expected_mean, rtol=1e-8)
    assert torch.allclose(std**2, expected_var, rtol=1e-8)


def test_posterior_recovers_the_ddpm_marginal_variance_on_vp():
    # Sanity on VP: the posterior variance is non-negative and the mean is a
    # genuine interpolation (bounded by the two inputs is not required, but
    # finiteness and the s->0 collapse are).
    process = VP()
    t = torch.full((8,), 0.5, dtype=torch.float64)
    s = torch.full((8,), 0.5 - 1e-6, dtype=torch.float64)
    x_t = torch.randn(8, 1, dtype=torch.float64)
    x0 = torch.randn(8, 1, dtype=torch.float64)
    mean, std = process.sde().posterior(x_t, x0, t, s)
    # as s -> t the posterior concentrates on x_t
    assert torch.allclose(mean, x_t, atol=1e-4)
    assert std.max().item() < 1e-2

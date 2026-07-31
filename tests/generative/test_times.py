import math

import pytest
import torch

from schnetpack.generative import (
    VE,
    VELinear,
    VP,
    Diffuse,
    LogNormalSigmaTimes,
    PseudoForceParametrization,
    UniformTimes,
)


@pytest.fixture
def ve():
    return VE(sigma_min=0.05, sigma_max=30.0)


# --- Process.t_of_sigma -------------------------------------------------- #


@pytest.mark.parametrize(
    "process",
    [VE(sigma_min=0.05, sigma_max=30.0), VELinear(), VP()],
    ids=["ve", "ve_linear", "vp"],
)
def test_t_of_sigma_inverts_sigma(process):
    """The inverse round-trips: sigma(t_of_sigma(s)) == s."""
    t = torch.linspace(process.t_min + 1e-3, process.t_max - 1e-3, 25)
    sigma = process.sigma(t)
    assert torch.allclose(process.t_of_sigma(sigma), t, atol=1e-4)


def test_ve_closed_form_matches_bisection(ve):
    """VE's analytic override agrees with the generic fallback."""
    sigma = torch.logspace(math.log10(0.05), math.log10(30.0), 40)
    from schnetpack.generative.processes import Process

    generic = Process.t_of_sigma(ve, sigma)
    assert torch.allclose(ve.t_of_sigma(sigma), generic, atol=1e-5)


def test_t_of_sigma_is_affine_in_log_sigma_on_ve(ve):
    """The property LogNormalSigmaTimes' normal-in-t result rests on."""
    sigma = torch.tensor([0.1, 1.0, 10.0])
    t = ve.t_of_sigma(sigma)
    steps = t[1:] - t[:-1]  # equal ratios in sigma -> equal steps in t
    assert torch.allclose(steps, steps[0].expand_as(steps), atol=1e-5)


def test_t_of_sigma_clamps_outside_the_schedule(ve):
    out = ve.t_of_sigma(torch.tensor([1e-6, 1e6]))
    assert out[0] == pytest.approx(ve.t_min)
    assert out[1] == pytest.approx(ve.t_max)


# --- UniformTimes -------------------------------------------------------- #


def test_uniform_times_matches_the_process_default(ve):
    sampler = UniformTimes(ve)
    torch.manual_seed(0)
    ours = sampler(10_000)
    torch.manual_seed(0)
    theirs = ve.sample_t(10_000)
    assert torch.allclose(ours, theirs)


def test_uniform_times_can_widen_the_range(ve):
    t = UniformTimes(ve, t_min=0.0, t_max=0.5)(5000)
    assert t.min() >= 0.0 and t.max() <= 0.5
    assert t.mean() == pytest.approx(0.25, abs=0.02)


# --- LogNormalSigmaTimes ------------------------------------------------- #


def test_lognormal_recovers_the_sigma_density():
    """The point of the class: the *sigma* it induces is the one asked for.

    Measured on a schedule wide enough that neither bound bites — on a
    narrower one the bounds are supposed to distort the tails, which is what
    test_truncate_rejects_instead_of_clamping pins down.
    """
    wide = VE(sigma_min=1e-4, sigma_max=1e4)
    torch.manual_seed(0)
    log_sigma = wide.sigma(LogNormalSigmaTimes(wide, mean=-0.7, std=1.2)(200_000)).log()
    assert log_sigma.mean() == pytest.approx(-0.7, abs=0.02)
    assert log_sigma.std() == pytest.approx(1.2, abs=0.02)


def test_lognormal_induces_a_normal_over_t_on_ve(ve):
    """t affine in log sigma => the times themselves are normal."""
    sampler = LogNormalSigmaTimes(ve, mean=-0.7, std=1.2)
    mean, std = sampler.induced_normal()
    assert mean == pytest.approx(0.3589, abs=1e-3)
    assert std == pytest.approx(0.1876, abs=1e-3)

    torch.manual_seed(0)
    t = sampler(200_000)
    interior = t[(t > ve.t_min + 1e-6) & (t < ve.t_max - 1e-6)]
    assert interior.mean() == pytest.approx(mean, abs=0.02)
    assert interior.std() == pytest.approx(std, abs=0.02)


def test_lognormal_concentrates_where_uniform_does_not(ve):
    """The reason to use it: far more mass in the band that decides geometry."""
    torch.manual_seed(0)
    lognormal = ve.sigma(LogNormalSigmaTimes(ve)(50_000))
    uniform = ve.sigma(UniformTimes(ve)(50_000))
    assert (lognormal < 1.0).float().mean() > 0.7
    assert (uniform < 1.0).float().mean() < 0.55


def test_truncate_rejects_instead_of_clamping(ve):
    """Both honour the bound; clamping piles mass on it, rejection does not."""
    torch.manual_seed(0)
    clamped = LogNormalSigmaTimes(ve, sigma_max=5.0)(20_000)
    truncated = LogNormalSigmaTimes(ve, sigma_max=5.0, truncate=True)(20_000)
    t_cap = ve.t_of_sigma(torch.tensor(5.0))

    assert clamped.max() <= t_cap + 1e-5
    assert truncated.max() <= t_cap + 1e-5
    at_cap = lambda t: ((t - t_cap).abs() < 1e-5).float().mean()
    assert at_cap(clamped) > 0.01  # the rejected tail lands exactly on it
    assert at_cap(truncated) == 0.0  # redrawn into the interior instead


def test_truncate_gives_up_on_an_impossible_range(ve):
    with pytest.raises(RuntimeError, match="almost never lands"):
        LogNormalSigmaTimes(
            ve, mean=-0.7, std=0.1, sigma_min=20.0, sigma_max=25.0, truncate=True
        )(64)


def test_lognormal_rejects_a_scaleless_prior():
    from schnetpack.generative.priors import Prior

    class Scaleless(Prior):
        std = None
        gaussian = False

        def sample(self, shape, dtype=None, device=None, context=None):
            return torch.zeros(*shape, dtype=dtype, device=device)

    with pytest.raises(ValueError, match="no scalar endpoint scale"):
        LogNormalSigmaTimes(VE(b_min=1e-3, prior=Scaleless()))


def test_bad_std_is_rejected(ve):
    with pytest.raises(ValueError, match="std must be positive"):
        LogNormalSigmaTimes(ve, std=0.0)


# --- integration with the consumers -------------------------------------- #


def test_diffuse_accepts_a_time_sampler(ve):
    """The transform route: one time per structure, drawn from our density."""
    sampler = LogNormalSigmaTimes(ve, mean=-0.7, std=1.2)
    diffuse = Diffuse(ve, PseudoForceParametrization(), t_sampler=sampler)

    torch.manual_seed(0)
    times = []
    for _ in range(400):
        out = diffuse({"_positions": torch.randn(7, 3)})
        assert out["t"].shape == (7,)
        assert torch.allclose(out["t"], out["t"][0])  # one time per structure
        times.append(out["t"][0])

    log_sigma = ve.sigma(torch.stack(times)).log()
    assert log_sigma.mean() == pytest.approx(-0.7, abs=0.2)


def test_matching_loss_accepts_a_time_sampler(ve):
    from schnetpack.generative import MatchingLoss

    seen = {}

    def spy(n, device=None):
        t = LogNormalSigmaTimes(ve)(n, device)
        seen["n"] = n
        return t

    loss_fn = MatchingLoss(ve, PseudoForceParametrization(), t_sampler=spy)
    loss = loss_fn(lambda x, t, cond=None: torch.zeros_like(x), torch.randn(16, 3))
    assert seen["n"] == 16  # one per sample, unlike Diffuse
    assert torch.isfinite(loss)

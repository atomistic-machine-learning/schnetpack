import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from schnetpack import properties
from schnetpack.generative import (
    VE,
    Diffuse,
    GaussianInterpolant,
    LogNormalTimes,
    NormalTimes,
    PseudoForceParametrization,
    UniformTimes,
    Variable,
    VELinear,
)


@pytest.fixture
def ve():
    return GaussianInterpolant(VE(sigma_min=0.05, sigma_max=30.0))


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


# --- NormalTimes and LogNormalTimes ------------------------------------ #


def truncated_moments(mean, std, lo, hi):
    """Mean and standard deviation of N(mean, std^2) truncated to [lo, hi]."""
    a, b = (lo - mean) / std, (hi - mean) / std

    def pdf(z):
        return 0.0 if math.isinf(z) else math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    def zpdf(z):
        return 0.0 if math.isinf(z) else z * pdf(z)

    mass = 0.5 * math.erfc(-b / math.sqrt(2)) - 0.5 * math.erfc(-a / math.sqrt(2))
    shift = (pdf(a) - pdf(b)) / mass
    var = std**2 * (1 + (zpdf(a) - zpdf(b)) / mass - shift**2)
    return mean + std * shift, math.sqrt(var)


def test_normal_times_is_the_truncated_normal(ve):
    """Both ends of the time range cut into this law; the draws follow it."""
    torch.manual_seed(0)
    t = NormalTimes(ve, mean=0.3, std=0.25)(400_000).double()
    mean, std = truncated_moments(0.3, 0.25, 0.0, 1.0)
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert float(t.mean()) == pytest.approx(mean, abs=0.003)
    assert float(t.std()) == pytest.approx(std, abs=0.003)


def test_lognormal_times_is_the_truncated_log_normal():
    process = GaussianInterpolant(VELinear(sigma_max=30.0, t_min=1e-3))
    torch.manual_seed(0)
    log_t = LogNormalTimes(process, mean=-3.0, std=1.2)(400_000).double().log()
    mean, std = truncated_moments(-3.0, 1.2, math.log(1e-3), 0.0)
    assert float(log_t.mean()) == pytest.approx(mean, abs=0.005)
    assert float(log_t.std()) == pytest.approx(std, abs=0.005)


def test_lognormal_times_on_velinear_is_the_edm_sigma_density():
    """sigma proportional to t: log sigma ~ N(-0.7, 1.2^2) up to sigma_max, GPFF's
    training law, from mean = -0.7 - log(sigma_max / t_max)."""
    process = GaussianInterpolant(VELinear(sigma_max=30.0, t_min=0.0))
    torch.manual_seed(0)
    t = LogNormalTimes(process, mean=-0.7 - math.log(30.0), std=1.2)(400_000)
    log_sigma = process.schedule.sigma(t.double()).log()
    mean, std = truncated_moments(-0.7, 1.2, -math.inf, math.log(30.0))
    assert float(log_sigma.max()) <= math.log(30.0) + 1e-6
    assert float(log_sigma.mean()) == pytest.approx(mean, abs=0.005)
    assert float(log_sigma.std()) == pytest.approx(std, abs=0.005)


def test_lognormal_times_concentrate_where_uniform_does_not():
    """The reason to use them: most draws in the band that decides geometry."""
    process = GaussianInterpolant(VELinear(sigma_max=30.0, t_min=0.0))
    torch.manual_seed(0)
    sigma = process.schedule.sigma
    gpff = sigma(LogNormalTimes(process, -0.7 - math.log(30.0), 1.2)(50_000))
    uniform = sigma(UniformTimes(process)(50_000))
    assert (gpff < 1.0).float().mean() > 0.7
    assert (uniform < 1.0).float().mean() < 0.05


def test_truncation_does_not_pile_mass_on_the_bounds(ve):
    """Draws past a bound are redrawn inside, not clamped onto it."""
    torch.manual_seed(0)
    t = NormalTimes(ve, mean=0.9, std=0.3, t_max=0.95)(20_000)
    assert t.max() <= 0.95
    assert ((t - 0.95).abs() < 1e-6).float().mean() == 0.0


class _CountOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("cls", [NormalTimes, LogNormalTimes])
def test_a_draw_is_a_few_operations_whatever_its_size(ve, cls):
    """One draw, not a search: on the GPU each operation is a kernel launch."""
    sampler = cls(ve, mean=-1.0 if cls is LogNormalTimes else 0.4, std=0.5)
    counts = []
    for n in (10, 10_000):
        with _CountOps() as ops:
            sampler(n)
        counts.append(ops.n)
    assert counts[0] == counts[1] <= 12


def test_an_impossible_range_raises(ve):
    with pytest.raises(RuntimeError, match="almost never lands"):
        NormalTimes(ve, mean=5.0, std=0.1)(64)


@pytest.mark.parametrize("cls", [NormalTimes, LogNormalTimes])
def test_bad_arguments_are_rejected(ve, cls):
    with pytest.raises(ValueError, match="std must be positive"):
        cls(ve, mean=0.0, std=0.0)
    with pytest.raises(ValueError, match="empty time range"):
        cls(ve, mean=0.0, std=1.0, t_min=0.6, t_max=0.4)


def test_lognormal_times_need_a_positive_t_max(ve):
    with pytest.raises(ValueError, match="t_max > 0"):
        LogNormalTimes(ve, mean=0.0, std=1.0, t_min=-1.0, t_max=0.0)


# --- integration with the consumers -------------------------------------- #


def test_diffuse_accepts_a_time_sampler(ve):
    """The transform route: one time per structure, drawn from our density."""
    sampler = NormalTimes(ve, mean=0.4, std=0.1)
    v = Variable(
        "positions", properties.R, "atom", ve, PseudoForceParametrization(), "y"
    )
    diffuse = Diffuse([v], t_sampler=sampler)

    torch.manual_seed(0)
    times = []
    for _ in range(400):
        out = diffuse({properties.R: torch.randn(7, 3)})
        assert out["t"].shape == (7,)
        assert torch.allclose(out["t"], out["t"][0])  # one time per structure
        times.append(out["t"][0])

    assert float(torch.stack(times).mean()) == pytest.approx(0.4, abs=0.02)

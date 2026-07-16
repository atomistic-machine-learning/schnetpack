import math

import pytest
import torch
from torch import nn

from schnetpack.generative import EDMPath, expand_t
from schnetpack.generative.parametrizations import X0Parametrization
from schnetpack.generative.preconditioning import (
    EDMPreconditioner,
    Preconditioner,
    PrecondDenoiser,
)


class RecordingNet(nn.Module):
    """Passes its input through and remembers what it was called with."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, cond=None):
        self.calls.append((x, t, cond))
        return x


# --- the trivial default -------------------------------------------------- #


def test_default_preconditioner_is_the_identity_wrapper():
    # c_skip=0, c_out=1, c_in=1 must leave the net untouched, so the base class
    # is genuinely "no preconditioning" rather than a special case.
    net = RecordingNet()
    denoiser = PrecondDenoiser(net)

    x = torch.randn(8, 3)
    t = torch.rand(8) + 0.1
    out = denoiser(x, t)

    assert torch.allclose(out, x)
    seen_x, seen_t, _ = net.calls[0]
    assert torch.allclose(seen_x, x)
    assert torch.allclose(seen_t, t)


def test_default_c_noise_is_finite_at_zero():
    # A log-based c_noise would diverge here; paths with t_min = 0 exist.
    p = Preconditioner()
    t = torch.tensor([0.0, 1e-12, 1.0])
    assert torch.isfinite(p.c_noise(t)).all()
    assert torch.equal(p.c_noise(t), t)


def test_default_scalings_have_the_shape_of_t():
    p = Preconditioner()
    t = torch.rand(6)
    for c in (p.c_skip(t), p.c_out(t), p.c_in(t), p.c_noise(t)):
        assert c.shape == t.shape


# --- EDM scalings --------------------------------------------------------- #


def test_edm_scalings_at_a_reference_sigma():
    p = EDMPreconditioner(sigma_data=0.5)
    t = torch.tensor([0.5])

    assert p.c_skip(t).item() == pytest.approx(0.5)
    assert p.c_out(t).item() == pytest.approx(0.25 / math.sqrt(0.5))
    assert p.c_in(t).item() == pytest.approx(1.0 / math.sqrt(0.5))
    assert p.c_noise(t).item() == pytest.approx(0.25 * math.log(0.5))


def test_edm_denoiser_approaches_the_identity_at_zero_noise():
    # c_skip -> 1 and c_out -> 0: with no noise there is nothing to denoise.
    p = EDMPreconditioner(sigma_data=0.5)
    t = torch.tensor([1e-6])
    assert p.c_skip(t).item() == pytest.approx(1.0, abs=1e-6)
    assert p.c_out(t).item() == pytest.approx(0.0, abs=1e-5)


def test_edm_scalings_normalize_at_large_noise():
    # c_in -> 0 kills the input scale and c_skip -> 0 drops the skip, so the
    # net is asked for the data statistics alone.
    p = EDMPreconditioner(sigma_data=0.5)
    t = torch.tensor([80.0])
    assert p.c_skip(t).item() == pytest.approx(0.0, abs=1e-3)
    assert p.c_in(t).item() == pytest.approx(1.0 / 80.0, rel=1e-3)
    assert p.c_out(t).item() == pytest.approx(0.5, rel=1e-3)


def test_edm_c_in_normalizes_the_input_variance():
    # c_in x should have unit variance for x = x0 + sigma eps with x0 of
    # scale sigma_data — the property c_in is derived from.
    torch.manual_seed(0)
    sigma_data = 0.5
    p = EDMPreconditioner(sigma_data=sigma_data)

    for sigma in [0.01, 0.5, 10.0, 80.0]:
        x0 = sigma_data * torch.randn(50000, 1)
        x = x0 + sigma * torch.randn(50000, 1)
        t = torch.full((50000,), sigma)
        scaled = x * expand_t(p.c_in(t), x)
        assert scaled.std().item() == pytest.approx(1.0, abs=0.02)


def test_edm_net_sees_scaled_input_and_compressed_noise_level():
    net = RecordingNet()
    precond = EDMPreconditioner(sigma_data=0.5)
    denoiser = PrecondDenoiser(net, precond)

    x = torch.randn(8, 3)
    t = torch.rand(8) * 10.0 + 0.1
    denoiser(x, t)

    seen_x, seen_t, _ = net.calls[0]
    assert torch.allclose(seen_x, x * expand_t(precond.c_in(t), x))
    assert torch.allclose(seen_t, precond.c_noise(t))
    assert seen_t.shape == t.shape  # per-sample, unexpanded


def test_cond_is_passed_to_the_net():
    net = RecordingNet()
    denoiser = PrecondDenoiser(net, EDMPreconditioner())
    marker = object()
    denoiser(torch.randn(4, 3), torch.rand(4) + 0.1, marker)
    assert net.calls[0][2] is marker


# --- full wiring ---------------------------------------------------------- #


def analytic_denoiser(x, sigma, mu0, s0):
    """E[x0 | x] for data N(mu0, s0^2) noised to x = x0 + sigma eps."""
    var = s0**2 + sigma**2
    return (s0**2 * x + sigma**2 * mu0) / var


class InvertingNet(nn.Module):
    """
    Emits the F that makes the preconditioned denoiser equal the analytic one.

    Recovers sigma from c_noise (sigma = exp(4 c_noise)) and x from c_in, then
    returns (D_target - c_skip x) / c_out. If the wrapper is wired correctly
    this composes back to exactly D_target.
    """

    def __init__(self, precond, mu0, s0):
        super().__init__()
        self.precond, self.mu0, self.s0 = precond, mu0, s0

    def forward(self, scaled_x, c_noise, cond=None):
        p = self.precond
        sigma = torch.exp(4.0 * c_noise)
        x = scaled_x / expand_t(p.c_in(sigma), scaled_x)
        target = analytic_denoiser(x, expand_t(sigma, x), self.mu0, self.s0)
        return (target - expand_t(p.c_skip(sigma), x) * x) / expand_t(p.c_out(sigma), x)


def test_preconditioned_denoiser_reproduces_the_analytic_denoiser():
    torch.manual_seed(0)
    mu0, s0 = 0.3, 0.5
    precond = EDMPreconditioner(sigma_data=s0)
    denoiser = PrecondDenoiser(InvertingNet(precond, mu0, s0), precond)

    x = torch.randn(64, 3, dtype=torch.float64) * 2.0
    sigma = torch.linspace(0.01, 20.0, 64, dtype=torch.float64)

    expected = analytic_denoiser(x, expand_t(sigma, x), mu0, s0)
    assert torch.allclose(denoiser(x, sigma), expected, rtol=1e-8, atol=1e-10)


def test_preconditioned_denoiser_composes_with_x0_parametrization():
    # The point of preconditioning being a net-to-net wrapper: downstream it is
    # an ordinary x0-denoiser, and the score follows by Tweedie.
    torch.manual_seed(0)
    mu0, s0 = 0.3, 0.5
    path = EDMPath()
    precond = EDMPreconditioner(sigma_data=s0)
    denoiser = PrecondDenoiser(InvertingNet(precond, mu0, s0), precond)
    parametrization = X0Parametrization(path)

    x = torch.randn(64, 3, dtype=torch.float64)
    sigma = torch.linspace(0.05, 20.0, 64, dtype=torch.float64)
    raw = denoiser(x, sigma)

    # alpha = 1 on EDMPath, so Tweedie reads score = (D - x) / sigma^2, which
    # for Gaussian data is the exact score of the marginal.
    score = parametrization.to_score(raw, x, sigma)
    expected = -(x - mu0) / expand_t(s0**2 + sigma**2, x)
    assert torch.allclose(score, expected, rtol=1e-6, atol=1e-8)


def test_precond_denoiser_registers_the_net_as_a_submodule():
    net = nn.Linear(3, 3)
    denoiser = PrecondDenoiser(net, EDMPreconditioner())
    assert list(denoiser.parameters()) == list(net.parameters())

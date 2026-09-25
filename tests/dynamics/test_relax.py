import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import DirectDenoising
from schnetpack.generative import (
    GaussianPrior,
    MatchingLoss,
    PCVarianceCoupling,
    PseudoForceParametrization,
    ScoreParametrization,
    VE,
    VP,
)

from tests.dynamics.test_sampling import IDLE, batch_model, draw, train_toy


# --- direct denoising ------------------------------------------------------ #


def test_direct_denoising_ends_on_the_x0_estimate():
    # The last iteration injects nothing (noise ratio 0) and then jumps, so a
    # model whose x0-estimate is a fixed point lands there exactly.
    mu0 = 1.5
    process = VE(0.01, 3.0)

    def model(x, t):
        return 2.0 * (mu0 - x)  # pseudo force straight to mu0

    sampler = DirectDenoising(batch_model(model), process, PseudoForceParametrization())

    out = draw(sampler, (16, 3), 5)
    assert torch.allclose(out, torch.full_like(out, mu0))


def test_direct_denoising_is_time_free_and_passes_the_batch():
    seen = []

    def model(batch):
        seen.append(batch)
        return {"prediction": torch.zeros_like(batch[properties.R])}

    process = VE(0.01, 3.0)
    sampler = DirectDenoising(model, process, PseudoForceParametrization())
    sampler.denoise({properties.R: torch.randn(4, 1), "condition": 7}, 3)

    assert len(seen) == 3
    assert all((b[properties.t] == 0.0).all() for b in seen)
    assert all(b["condition"] == 7 for b in seen)


def test_direct_denoising_lambda_zero_is_deterministic():
    process = VE(0.01, 3.0)
    model = batch_model(lambda x, t: -x)  # some deterministic field
    sampler = DirectDenoising(
        model, process, PseudoForceParametrization(), stochastic_lambda=0.0
    )

    batch = {properties.R: torch.randn(8, 2)}
    out1 = sampler.denoise(batch, 10)
    out2 = sampler.denoise(batch, 10)
    assert torch.equal(out1[properties.R], out2[properties.R])


def test_direct_denoising_validates_pair_and_prior():
    # Same construction contract as Sampler: the pair is validated, and a
    # marginal-changing coupling has no data-free start to derive.
    reshaped = VP(coupling=PCVarianceCoupling())
    with pytest.raises(TypeError, match="Gaussian kernel"):
        DirectDenoising(IDLE, reshaped, ScoreParametrization(), prior=GaussianPrior())
    with pytest.raises(ValueError, match="marginal"):
        DirectDenoising(IDLE, reshaped, PseudoForceParametrization())

    explicit = GaussianPrior()
    sampler = DirectDenoising(
        IDLE, reshaped, PseudoForceParametrization(), prior=explicit
    )
    assert sampler.prior is explicit


class TimeFreeToyNet(torch.nn.Module):
    """An MLP on x alone — the time-free contract direct denoising presumes."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x, t, cond=None):
        return self.layers(x)


def test_direct_denoising_trained_gpff_assembly():
    # The full GPFF recipe end to end: scaled VE + pseudo-force head with the
    # clipped 1/b^2 weight, a time-free net, and the stochastic
    # direct-denoising loop.
    torch.manual_seed(0)
    mu, sd = 1.0, 0.5
    process = VE(0.01, 3.0)
    parametrization = PseudoForceParametrization()
    loss = MatchingLoss(
        process,
        parametrization,
        weight=lambda t: (1.0 / process.b(t) ** 2).clamp(max=1.0),
    )
    model = train_toy(loss, TimeFreeToyNet(), mu, sd)

    sampler = DirectDenoising(
        batch_model(model), process, parametrization, stochastic_lambda=1.0
    )
    samples = draw(sampler, (4096, 1), 50)

    assert torch.isfinite(samples).all()
    assert samples.mean().item() == pytest.approx(mu, abs=0.2)

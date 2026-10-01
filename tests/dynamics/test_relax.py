import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import DirectDenoising, ForceCalculator, GenerativeCalculator
from schnetpack.generative import (
    VE,
    DatasetPrior,
    MatchingLoss,
    PseudoForceParametrization,
)
from tests.dynamics.test_sampling import IDLE, batch_model, draw, train_toy

# --- direct denoising ------------------------------------------------------ #


def gpff(model, process=None, **kwargs):
    """Direct denoising on a pseudo-force model, started from the process's prior."""
    process = process if process is not None else VE(0.01, 3.0)
    return DirectDenoising(
        ForceCalculator(model, kind="pseudo"),
        prior=process.sampling_prior(),
        **kwargs,
    )


def test_direct_denoising_ends_on_the_x0_estimate():
    # A model whose x0-estimate is a fixed point lands there exactly.
    mu0 = 1.5
    process = VE(0.01, 3.0)

    def model(x, t):
        return 2.0 * (mu0 - x)  # pseudo force straight to mu0

    sampler = gpff(batch_model(model), process)

    out = draw(sampler, (16, 3), 5)
    assert torch.allclose(out, torch.full_like(out, mu0))


def test_direct_denoising_is_time_free_and_passes_the_batch():
    seen = []

    def model(batch):
        seen.append(batch)
        return {"prediction": torch.zeros_like(batch[properties.R])}

    sampler = gpff(model)
    sampler.run({properties.R: torch.randn(4, 1), "condition": 7}, 3)

    assert len(seen) == 3
    assert all((b[properties.t] == 0.0).all() for b in seen)
    assert all(b["condition"] == 7 for b in seen)


def test_direct_denoising_relaxes_structures_from_a_dataset_prior():
    # The relaxation start: stored non-equilibrium structures, positions as
    # they are, handed to the loop by sample().
    dataset = [
        {
            properties.Z: torch.tensor([1, 8, 1]),
            properties.R: torch.full((3, 3), 5.0),
            properties.n_atoms: torch.tensor([3]),
        }
    ]
    seen = []

    def model(batch):
        seen.append(batch[properties.R].clone())
        return {"prediction": torch.zeros_like(batch[properties.R])}

    sampler = DirectDenoising(
        ForceCalculator(model, kind="pseudo"),
        prior=DatasetPrior(dataset),
    )
    out = sampler.sample(2, n_steps=1)

    assert torch.equal(seen[0], torch.full((6, 3), 5.0))
    assert torch.equal(out[properties.Z], torch.tensor([1, 8, 1, 1, 8, 1]))


def test_direct_denoising_is_deterministic():
    model = batch_model(lambda x, t: -x)  # some deterministic field
    sampler = gpff(model)

    batch = {properties.R: torch.randn(8, 2)}
    out1 = sampler.run(batch, 10)
    out2 = sampler.run(batch, 10)
    assert torch.equal(out1[properties.R], out2[properties.R])


def test_direct_denoising_runs_on_a_pseudo_force_only():
    # a bare model is taken as a pseudo-force in Angstrom
    calculator = DirectDenoising(IDLE).calculator
    assert isinstance(calculator, ForceCalculator) and not calculator.physical
    with pytest.raises(TypeError, match="physical"):
        DirectDenoising(ForceCalculator(IDLE))
    with pytest.raises(TypeError, match="ForceCalculator"):
        DirectDenoising(
            GenerativeCalculator(IDLE, VE(0.01, 3.0), PseudoForceParametrization())
        )


def test_direct_denoising_stops_on_fmax():
    # F = 2 (0 - x): the first jump lands on 0, the second check stops the run
    calls = []

    def model(batch):
        calls.append(None)
        return {"prediction": -2.0 * batch[properties.R]}

    batch = {
        properties.R: torch.randn(3, 3),
        properties.n_atoms: torch.tensor([3]),
        properties.idx_m: torch.zeros(3, dtype=torch.long),
    }
    out = gpff(model, fmax=1e-6).run(batch, 10)
    assert torch.equal(out[properties.R], torch.zeros(3, 3))
    assert len(calls) == 2


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
    # clipped 1/b^2 weight, a time-free net, and the direct-denoising loop.
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

    sampler = gpff(batch_model(model), process)
    samples = draw(sampler, (4096, 1), 50)

    assert torch.isfinite(samples).all()
    assert samples.mean().item() == pytest.approx(mu, abs=0.2)

"""Diffuse and VariableOutput on collated and padded batches."""

import math

import pytest
import torch

import schnetpack.properties as properties
from schnetpack.generative import (
    VE,
    Diffuse,
    GaussianInterpolant,
    GaussianPrior,
    LogNormalSigmaTimes,
    PseudoForceParametrization,
    Variable,
    VariableOutput,
    VELinear,
)
from schnetpack.objectives import calculate_loss, extract_targets


def gpff_positions():
    noising = GaussianInterpolant(
        VELinear(sigma_max=30.0, t_min=0.0),
        prior=GaussianPrior(std=30.0, centered=False),
    )
    return Variable(
        name="positions",
        view=properties.R,
        per="atom",
        noising=noising,
        parametrization=PseudoForceParametrization(),
        output_key="pseudo_force_pred",
        weight=lambda schedule, t, x_t: (1.0 / schedule.sigma(t) ** 2).clamp(max=1.0),
    )


def padded_batch(n_atoms=(3, 5, 2), n_pad=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = torch.tensor(list(n_atoms) + [n_pad])
    N = int(n.sum())
    real = int(n[:-1].sum())
    return {
        properties.Z: torch.randint(1, 10, (N,), generator=g),
        properties.R: torch.randn(N, 3, generator=g, dtype=torch.float64),
        properties.idx_m: torch.arange(len(n)).repeat_interleave(n),
        properties.n_atoms: n,
        properties.atom_mask: (torch.arange(N) < real).double(),
        properties.structure_mask: (torch.arange(len(n)) < len(n) - 1).double(),
    }


def test_one_time_per_structure_and_no_weight_on_padding():
    v = gpff_positions()
    times = LogNormalSigmaTimes(v.noising, truncate=True)
    batch = Diffuse([v], t_sampler=times)(padded_batch())
    t, idx_m = batch[properties.t], batch[properties.idx_m]
    for m in range(4):
        assert t[idx_m == m].unique().numel() == 1
    assert torch.equal(batch["t_structure"][idx_m], t)
    mask = batch[properties.atom_mask]
    assert (batch[v.weight_key][mask == 0] == 0).all()
    sigma = v.noising.schedule.sigma(t)
    expected = (1.0 / sigma**2).clamp(max=1.0) * mask
    torch.testing.assert_close(batch[v.weight_key], expected)
    # the label is the parametrization's target: 2 (x0 - x_t)
    x0 = padded_batch()[properties.R]
    torch.testing.assert_close(batch[v.label_key], 2.0 * (x0 - batch[properties.R]))


def test_variable_output_ignores_padding():
    v = gpff_positions()
    output = VariableOutput(v)
    batch = Diffuse([v])(padded_batch())
    pred = {v.output_key: torch.randn_like(batch[properties.R])}
    targets = extract_targets([output], batch)
    assert v.weight_key in targets and properties.atom_mask in targets
    loss = calculate_loss([output], pred, targets)

    keep = batch[properties.atom_mask].bool()
    w = batch[v.weight_key][keep]
    err = (pred[v.output_key][keep] - batch[v.label_key][keep]) ** 2
    torch.testing.assert_close(loss, (w[:, None] * err).sum() / err.numel())
    # without padding rows and mask, the same value
    unpadded = {v.label_key: batch[v.label_key][keep], v.weight_key: w}
    loss2 = output.calculate_loss({v.output_key: pred[v.output_key][keep]}, unpadded)
    torch.testing.assert_close(loss2, loss)


def test_truncated_lognormal_is_the_truncated_law():
    """Inverse-CDF sampling: the moments of the truncated normal in log sigma."""
    process = GaussianInterpolant(VE(sigma_min=1e-3, sigma_max=2.0))
    torch.manual_seed(0)
    t = LogNormalSigmaTimes(process, mean=-0.7, std=1.2, truncate=True)(400_000)
    x = process.schedule.sigma(t).double().log()
    a = (math.log(1e-3) + 0.7) / 1.2
    b = (math.log(2.0) + 0.7) / 1.2

    def pdf(z):
        return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    def cdf(z):
        return 0.5 * math.erfc(-z / math.sqrt(2))

    mass = cdf(b) - cdf(a)
    mean = -0.7 + 1.2 * (pdf(a) - pdf(b)) / mass
    var = 1.2**2 * (
        1 + (a * pdf(a) - b * pdf(b)) / mass - ((pdf(a) - pdf(b)) / mass) ** 2
    )
    assert x.max() <= math.log(2.0) + 1e-6 and x.min() >= math.log(1e-3) - 1e-6
    assert float(x.mean()) == pytest.approx(mean, abs=0.005)
    assert float(x.std()) == pytest.approx(math.sqrt(var), abs=0.005)

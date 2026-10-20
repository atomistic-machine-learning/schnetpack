"""
Variables, and the priors and interpolants that draw under a variable's key.
"""

import pytest
import torch

from schnetpack import properties
from schnetpack.generative import (
    VE,
    VP,
    EpsParametrization,
    GaussianInterpolant,
    GaussianPrior,
    Prior,
    Variable,
    noise_like,
    weighted_squared_error,
)

SIGMA_MAX = 10.0


def positions(centered=True):
    schedule = VE(sigma_min=0.01, sigma_max=SIGMA_MAX)
    return Variable(
        name="positions",
        key=properties.R,
        per="atom",
        noising=GaussianInterpolant(
            schedule, prior=GaussianPrior(std=SIGMA_MAX, centered=centered)
        ),
        parametrization=EpsParametrization(),
        output_key="eps_R",
    )


def cell_like(decode=None):
    """A per-structure variable of width 6 on VP, like the whitened cell."""
    return Variable(
        name="cell",
        key="cell_y",
        per="structure",
        noising=GaussianInterpolant(
            VP(),
            prior=GaussianPrior(
                std=1.0, centered=False, key="cell_y", per="structure", dim=6
            ),
        ),
        parametrization=EpsParametrization(),
        output_key="eps_cell",
        decode=decode,
    )


# --- priors on their own key ------------------------------------------------ #


def test_prior_draws_its_own_key_with_its_own_layout():
    prior = GaussianPrior(std=2.0, centered=False, key="cell_y", per="structure", dim=6)
    batch = {properties.n_atoms: torch.tensor([3, 4]), properties.R: torch.zeros(7, 3)}
    out = prior(batch)
    assert out["cell_y"].shape == (2, 6)
    assert out[properties.R] is batch[properties.R]
    assert (prior.key, prior.keys) == ("cell_y", ("cell_y",))
    assert prior.gaussian and prior.std == 2.0


def test_prior_draws_positions_from_the_layout():
    batch = {
        properties.n_atoms: torch.tensor([3, 4]),
        properties.idx_m: torch.tensor([0, 0, 0, 1, 1, 1, 1]),
    }
    x = GaussianPrior(std=SIGMA_MAX)(batch)[properties.R]
    assert x.shape == (7, 3)
    for m in range(2):
        assert x[batch[properties.idx_m] == m].mean(0).abs().max() < 1e-5


def test_prior_refuses_an_unknown_row_layout():
    with pytest.raises(ValueError, match="per must be"):
        GaussianPrior(per="molecule")


def test_prior_without_a_layout_names_what_it_needs():
    with pytest.raises(KeyError, match="cannot infer the shape of 'cell_y'"):
        GaussianPrior(key="cell_y", per="structure")({})


def test_noise_is_centered_per_structure_when_the_prior_is():
    idx_m = torch.tensor([0, 0, 0, 1, 1])
    x = torch.zeros(5, 3, dtype=torch.float64)
    noise = GaussianPrior().noise_like(x, {properties.idx_m: idx_m})
    for m in range(2):
        assert torch.allclose(noise[idx_m == m].mean(0), torch.zeros(3, dtype=x.dtype))


def test_uncentered_noise_is_randn_like():
    x = torch.zeros(5, 3, dtype=torch.float64)
    torch.manual_seed(5)
    noise = GaussianPrior(centered=False).noise_like(x, {})
    torch.manual_seed(5)
    assert torch.equal(noise, torch.randn_like(x))


def test_noise_like_falls_back_to_randn_like_for_other_priors():
    class Plain(Prior):
        def __call__(self, batch):
            return batch

    x = torch.zeros(5, 3)
    torch.manual_seed(5)
    noise = noise_like(Plain(), x, {})
    torch.manual_seed(5)
    assert torch.equal(noise, torch.randn_like(x))


# --- interpolants read the draw back under the prior's key -------------------- #


def test_interpolant_draws_x1_under_the_priors_key():
    variable = cell_like()
    inputs = {properties.R: torch.zeros(5, 3), "cell_y": torch.zeros(1, 6)}
    t = torch.tensor([0.5])
    x_t, x0, x1, _ = variable.noising.perturb(inputs["cell_y"], t=t, batch=inputs)
    assert x1.shape == (1, 6)
    assert torch.allclose(x_t, variable.noising.schedule.sigma(t)[:, None] * x1)


# --- variables ------------------------------------------------------------- #


def test_variable_keys_follow_its_name_and_rows():
    v = positions()
    assert (v.label_key, v.weight_key, v.clean_key, v.time_key) == (
        "label_positions",
        "weight_positions",
        "clean_positions",
        properties.t,
    )
    assert cell_like().time_key == "t_structure"


def test_variable_refuses_an_unknown_row_layout():
    with pytest.raises(ValueError, match="per must be"):
        Variable(
            "x",
            properties.R,
            "molecule",
            positions().noising,
            EpsParametrization(),
            "y",
        )


def test_variable_validates_its_parametrization():
    from schnetpack.generative import FlowMatching, Interpolant

    with pytest.raises(TypeError, match="Gaussian kernel"):
        Variable(
            "x",
            properties.R,
            "atom",
            Interpolant(FlowMatching()),
            EpsParametrization(),
            "y",
        )


def test_a_variable_and_its_prior_share_the_key():
    with pytest.raises(ValueError, match="give the prior key='cell_y'"):
        Variable(
            "cell",
            "cell_y",
            "structure",
            GaussianInterpolant(VP()),
            EpsParametrization(),
            "eps_cell",
        )


def test_loss_on_positions_is_three_times_the_mean_squared_error():
    v = positions()
    prediction, label = torch.randn(6, 3), torch.randn(6, 3)
    batch = {v.label_key: label, v.weight_key: torch.ones(6)}
    loss = v.loss({"eps_R": prediction}, batch)
    assert torch.allclose(loss, 3.0 * ((prediction - label) ** 2).mean())


def test_loss_weights_rows():
    prediction, label = torch.randn(4, 2), torch.zeros(4, 2)
    weight = torch.tensor([1.0, 0.0, 2.0, 0.0])
    per_row = (prediction**2).sum(1)
    expected = (per_row[0] + 2.0 * per_row[2]) / 4.0
    assert torch.allclose(weighted_squared_error(prediction, label, weight), expected)


def test_loss_uses_a_parametrizations_own_loss():
    class OwnLoss(EpsParametrization):
        def loss(self, prediction, label, weight):
            return torch.tensor(42.0)

    v = Variable("x", properties.R, "atom", positions().noising, OwnLoss(), "eps_R")
    batch = {v.label_key: torch.zeros(2, 3), v.weight_key: torch.ones(2)}
    assert v.loss({"eps_R": torch.ones(2, 3)}, batch) == 42.0


def test_write_runs_the_decode():
    def decode(batch):
        return {**batch, "decoded": batch["cell_y"] + 1.0}

    out = cell_like(decode=decode).write({}, torch.zeros(1, 6))
    assert torch.equal(out["decoded"], torch.ones(1, 6))

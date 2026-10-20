"""
Masked atom types: the masking, the logits head read as p(z0 | z_t), its
loss and weight, and a Diffuse batch at t = 1 as the sampling start.
"""

import pytest
import torch

from schnetpack import properties
from schnetpack.generative import (
    MASK,
    VE,
    Diffuse,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    LogitsParametrization,
    MaskedTypeWeight,
    MaskNoising,
    MaskPrior,
    ParametrizedModel,
    Variable,
)

T = 1000
N_CLASSES = 10


def types(n_classes=N_CLASSES):
    return Variable(
        name="types",
        key=properties.Z,
        per="atom",
        noising=MaskNoising(FlowMatching(t_min=1.0 / T, t_max=1.0)),
        parametrization=LogitsParametrization(n_classes=n_classes),
        weight=MaskedTypeWeight(n_steps=T),
        output_key="type_logits",
    )


def positions():
    return Variable(
        name="positions",
        key=properties.R,
        per="atom",
        noising=GaussianInterpolant(
            VE(sigma_min=0.01, sigma_max=10.0), prior=GaussianPrior(std=10.0)
        ),
        parametrization=EpsParametrization(),
        output_key="eps_R",
    )


# --- masking ------------------------------------------------------------------ #


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_masking_masks_a_fraction_t_and_keeps_the_rest(t):
    torch.manual_seed(0)
    z0 = torch.randint(1, N_CLASSES, (20000,))
    z_t, x0, x1, t_out = MaskNoising(FlowMatching()).perturb(
        z0, torch.full((20000,), t)
    )
    masked = z_t == MASK
    assert masked.float().mean().item() == pytest.approx(t, abs=0.015)
    assert torch.equal(z_t[~masked], z0[~masked])
    assert torch.equal(x0, z0) and x1 is None
    assert torch.equal(t_out, torch.full((20000,), t))


def test_masking_ends_at_every_atom_masked():
    z0 = torch.randint(1, N_CLASSES, (50,))
    z_t, *_ = MaskNoising(FlowMatching(t_max=1.0)).perturb(z0, torch.ones(50))
    assert (z_t == MASK).all()


def test_the_mask_prior_masks_every_atom():
    batch = {properties.Z: torch.tensor([1, 6, 8])}
    assert MaskPrior()(batch)[properties.Z].tolist() == [MASK] * 3
    batch = {properties.n_atoms: torch.tensor([2, 3])}
    z = MaskPrior()(batch)[properties.Z]
    assert z.dtype == torch.long and z.tolist() == [MASK] * 5
    assert MaskNoising(FlowMatching()).sampling_prior().key == properties.Z


# --- the logits head and its weight ---------------------------------------------- #


def test_the_weight_is_zero_on_revealed_atoms_and_one_step_kl_on_masked():
    weight = MaskedTypeWeight(n_steps=T, lambda_ce=0.005)
    t = torch.tensor([0.5, 0.5, 0.01, 1.0], dtype=torch.float64)
    z_t = torch.tensor([3, MASK, MASK, MASK])
    w = weight(FlowMatching(), t, z_t)
    assert w[0] == 0.0
    assert torch.allclose(w[1:], 1.0 / (T * t[1:]) + 0.005)


def test_x0_is_one_hot_on_revealed_atoms_and_never_mask():
    torch.manual_seed(0)
    param = LogitsParametrization(N_CLASSES)
    logits = torch.randn(4, N_CLASSES)
    logits[:, MASK] = 50.0  # ignored: no clean type is masked
    z_t = torch.tensor([MASK, 4, MASK, 7])
    probs = param.to_x0(FlowMatching(), logits, z_t, torch.full((4,), 0.5))
    assert torch.allclose(probs.sum(-1), torch.ones(4))
    assert (probs[:, MASK] == 0).all()
    assert torch.equal(probs[1], torch.eye(N_CLASSES)[4])
    assert torch.equal(probs[3], torch.eye(N_CLASSES)[7])
    expected = logits[0, 1:].softmax(-1)
    assert torch.allclose(probs[0, 1:], expected)


def test_the_loss_averages_the_weighted_ce_over_the_counted_atoms():
    torch.manual_seed(0)
    param = LogitsParametrization(N_CLASSES)
    logits = torch.randn(4, N_CLASSES)
    label = torch.tensor([2, 3, 4, 5])
    weight = torch.tensor([0.0, 2.0, 0.5, 0.0])
    log_p = logits[:, 1:].log_softmax(-1)  # MASK left out
    ce = -log_p[torch.arange(4), label - 1]
    expected = (2.0 * ce[1] + 0.5 * ce[2]) / 2
    assert torch.allclose(param.loss(logits, label, weight), expected)
    # nothing counted: zero, not nan
    assert param.loss(logits, label, torch.zeros(4)) == 0.0


def test_a_masked_type_has_no_score():
    param = LogitsParametrization(N_CLASSES)
    z_t = torch.tensor([MASK, 2])
    with pytest.raises(TypeError, match="no score"):
        param.to_score(FlowMatching(), torch.zeros(2, N_CLASSES), z_t, torch.ones(2))
    with pytest.raises(TypeError, match="no score"):
        param.to_pseudo_force(
            FlowMatching(), torch.zeros(2, N_CLASSES), z_t, torch.ones(2)
        )


def test_a_parametrized_model_reads_the_logits_as_x0():
    v = types()
    model = ParametrizedModel(
        lambda batch: {"type_logits": torch.zeros(3, N_CLASSES)},
        v.parametrization,
        v.noising.schedule,
        output_key="type_logits",
        key=properties.Z,
        x0_key="x0_Z",
    )
    batch = {properties.Z: torch.tensor([MASK, 5, MASK]), properties.t: torch.ones(3)}
    probs = model(batch)["x0_Z"]
    uniform = torch.full((N_CLASSES - 1,), 1.0 / (N_CLASSES - 1))
    assert torch.allclose(probs[0, 1:], uniform)
    assert probs[1, 5] == 1.0


# --- Diffuse ------------------------------------------------------------------------ #


def structure(n_atoms=6):
    return {
        properties.Z: torch.randint(1, N_CLASSES, (n_atoms,)),
        properties.R: torch.randn(n_atoms, 3, dtype=torch.float64),
        properties.n_atoms: torch.tensor([n_atoms]),
    }


def test_diffuse_writes_label_and_weight_of_the_types():
    torch.manual_seed(0)
    inputs = structure()
    z0 = inputs[properties.Z].clone()
    out = Diffuse([types(), positions()], t_sampler=lambda n, d: torch.full((n,), 0.5))(
        inputs
    )
    masked = out[properties.Z] == MASK
    assert torch.equal(out["label_types"], z0)
    assert torch.equal(out["weight_types"] > 0, masked)
    assert out[properties.t].shape == (6,)
    assert torch.equal(out[properties.Z][~masked], z0[~masked])


def test_at_t_1_a_diffuse_batch_matches_the_sampling_start():
    torch.manual_seed(0)
    t_types, t_pos = types(), positions()
    out = Diffuse([t_types, t_pos], t_sampler=lambda n, d: torch.ones(n))(
        structure(400)
    )
    start = {properties.n_atoms: torch.tensor([400])}
    for v in (t_types, t_pos):
        start = v.noising.sampling_prior()(start)
    assert (out[properties.Z] == MASK).all()
    assert torch.equal(out[properties.Z], start[properties.Z])
    # x0 + sigma_max eps with |x0| ~ 1 is close to the centered prior's spread
    assert out[properties.R].std().item() == pytest.approx(10.0, rel=0.1)
    assert start[properties.R].std().item() == pytest.approx(10.0, rel=0.1)


def test_the_types_loss_runs_through_the_variable():
    torch.manual_seed(0)
    v = types()
    out = Diffuse([v], t_sampler=lambda n, d: torch.full((n,), 0.7))(structure(50))
    logits = torch.randn(50, N_CLASSES, requires_grad=True)
    loss = v.loss({"type_logits": logits}, out)
    loss.backward()
    revealed = out[properties.Z] != MASK
    assert loss.item() > 0
    assert (logits.grad[revealed] == 0).all()
    assert (logits.grad[~revealed] != 0).any()

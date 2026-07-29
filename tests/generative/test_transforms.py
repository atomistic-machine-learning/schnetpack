import pytest
import torch

from schnetpack import properties
from schnetpack.generative import (
    Diffuse,
    EpsParametrization,
    FlowMatching,
    GaussianPrior,
    PCVarianceCoupling,
    PermutationCoupling,
    ScoreParametrization,
    VE,
    VelocityParametrization,
    VP,
    X0Parametrization,
    expand_t,
)


def structure(n_atoms=7):
    """A single pre-collation structure, as a dataloader worker would see it."""
    return {
        properties.Z: torch.randint(1, 9, (n_atoms,)),
        properties.R: torch.randn(n_atoms, 3),
        properties.n_atoms: torch.tensor([n_atoms]),
        properties.cell: torch.zeros(1, 3, 3),
        properties.pbc: torch.zeros(1, 3, dtype=torch.bool),
    }


@pytest.fixture
def vp():
    return VP()


# --- what it writes ------------------------------------------------------- #


def test_writes_x_t_label_and_times(vp):
    torch.manual_seed(0)
    inputs = structure(7)
    x0 = inputs[properties.R].clone()

    out = Diffuse(vp, EpsParametrization())(inputs)

    assert out[properties.R].shape == x0.shape
    assert not torch.allclose(out[properties.R], x0)  # positions were noised
    assert out["label"].shape == x0.shape
    assert out["t"].shape == (7,)  # per atom, for conditioning
    assert out["t_structure"].shape == (1,)  # per structure, for a time head


def test_times_agree_across_granularities(vp):
    torch.manual_seed(0)
    out = Diffuse(vp, EpsParametrization())(structure(5))
    # one time per structure, broadcast — not five independent draws
    assert torch.allclose(out["t"], out["t_structure"].expand(5))
    assert out["t"].unique().numel() == 1


def test_time_lands_in_the_processs_usable_range(vp):
    torch.manual_seed(0)
    times = [
        Diffuse(vp, EpsParametrization())(structure())["t_structure"].item()
        for _ in range(200)
    ]
    assert min(times) >= vp.t_min
    assert max(times) <= vp.t_max
    assert min(times) < 0.2 and max(times) > 0.8  # covers the range


def test_keeps_the_clean_structure_when_asked(vp):
    torch.manual_seed(0)
    inputs = structure()
    x0 = inputs[properties.R].clone()
    out = Diffuse(vp, EpsParametrization(), original_key="x_0")(inputs)
    assert torch.allclose(out["x_0"], x0)


def test_optional_keys_are_skipped_when_none(vp):
    out = Diffuse(vp, EpsParametrization(), structure_time_key=None)(structure())
    assert "t_structure" not in out
    assert "x_0" not in out


def test_untouched_properties_survive(vp):
    inputs = structure(6)
    z = inputs[properties.Z].clone()
    out = Diffuse(vp, EpsParametrization())(inputs)
    assert torch.equal(out[properties.Z], z)
    assert torch.equal(out[properties.n_atoms], torch.tensor([6]))


# --- the label is the parametrization's ----------------------------------- #


@pytest.mark.parametrize(
    "param_cls",
    [
        ScoreParametrization,
        EpsParametrization,
        X0Parametrization,
        VelocityParametrization,
    ],
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    "process_cls", [VP, VE, FlowMatching], ids=lambda c: c.__name__
)
def test_label_matches_the_parametrizations_target(process_cls, param_cls):
    # The whole point of "arbitrary label": swapping the parametrization must
    # swap the target with no other change.
    torch.manual_seed(0)
    process = process_cls()
    parametrization = param_cls()
    inputs = structure(6)
    x0 = inputs[properties.R].clone()

    out = Diffuse(process, parametrization)(inputs)

    # recover the noise the transform drew, then rebuild the target by hand
    t = out["t"]
    a = expand_t(process.a(t), x0)
    b = expand_t(process.b(t), x0)
    x1 = (out[properties.R] - a * x0) / b

    expected = parametrization.target(process, x0, x1, t)
    assert torch.allclose(out["label"], expected, rtol=1e-4, atol=1e-5)


def test_eps_label_is_the_noise_that_made_x_t(vp):
    # The concrete case: label == (x_t - a x0)/b.
    torch.manual_seed(0)
    inputs = structure(6)
    x0 = inputs[properties.R].clone()
    out = Diffuse(vp, EpsParametrization(), original_key="x_0")(inputs)

    t = out["t"]
    a = expand_t(vp.a(t), x0)
    b = expand_t(vp.b(t), x0)
    reconstructed = a * x0 + b * out["label"]
    assert torch.allclose(reconstructed, out[properties.R], rtol=1e-5, atol=1e-6)


def test_x0_label_is_the_clean_structure(vp):
    torch.manual_seed(0)
    inputs = structure()
    x0 = inputs[properties.R].clone()
    out = Diffuse(vp, X0Parametrization())(inputs)
    assert torch.allclose(out["label"], x0)


# --- the axes stay swappable ---------------------------------------------- #


@pytest.mark.parametrize(
    "process", [VP(), VE(), FlowMatching()], ids=lambda p: type(p).__name__
)
def test_any_schedule_works(process):
    torch.manual_seed(0)
    out = Diffuse(process, EpsParametrization())(structure())
    assert torch.isfinite(out[properties.R]).all()
    assert torch.isfinite(out["label"]).all()


def test_prior_decides_the_noise():
    # The extension point for constrained noise is the prior, not the
    # transform: a prior that projects out the mean must give a mean-free
    # label and a mean-free displacement, with no change to the transform.
    class MeanFreePrior(GaussianPrior):
        def sample_like(self, x0, context=None):
            z = super().sample_like(x0, context)
            return z - z.mean(0, keepdim=True)

    torch.manual_seed(0)
    process = VE(prior=MeanFreePrior())
    out = Diffuse(process, VelocityParametrization())(structure(8))

    # velocity target on VE is b_dot * x1, mean-free because x1 is
    assert out["label"].mean(0).abs().max() < 1e-6


def test_reconfigured_process_flows_through_the_transform():
    # A re-paired assembly diffuses just as the plain one does, as long as its
    # parametrization is valid — the transform validates the pair but never
    # inspects the process beyond that.
    process = VE(coupling=PermutationCoupling())
    out = Diffuse(process, VelocityParametrization())(structure(8))
    assert torch.isfinite(out["label"]).all()


def test_transform_validates_the_pair():
    # A Gaussian-only head on a kernel-less configuration must fail at
    # construction, not at the first batch.
    process = VP(coupling=PCVarianceCoupling())
    with pytest.raises(TypeError, match="Gaussian kernel"):
        Diffuse(process, EpsParametrization())


def test_time_sampler_hook_is_used(vp):
    out = Diffuse(
        vp,
        EpsParametrization(),
        t_sampler=lambda n, device: torch.full((n,), 0.42, device=device),
    )(structure())
    assert out["t_structure"].item() == pytest.approx(0.42)
    assert torch.allclose(out["t"], torch.full((7,), 0.42))


def test_custom_keys(vp):
    out = Diffuse(
        vp,
        EpsParametrization(),
        label_key="eps",
        time_key="time",
        structure_time_key="time_mol",
    )(structure())
    assert "eps" in out and "time" in out and "time_mol" in out
    assert "label" not in out and "t" not in out


def test_can_diffuse_a_property_other_than_positions(vp):
    torch.manual_seed(0)
    inputs = structure()
    inputs["velocities"] = torch.randn(7, 3)
    positions = inputs[properties.R].clone()

    out = Diffuse(vp, EpsParametrization(), diffuse_property="velocities")(inputs)

    assert not torch.allclose(out["velocities"], torch.zeros(7, 3))
    assert torch.equal(out[properties.R], positions)  # positions untouched


# --- shape and dtype ------------------------------------------------------ #


def test_time_matches_the_property_dtype(vp):
    inputs = structure()
    inputs[properties.R] = inputs[properties.R].double()
    out = Diffuse(vp, EpsParametrization())(inputs)
    assert out["t"].dtype == torch.float64
    assert out[properties.R].dtype == torch.float64


def test_is_a_preprocessor(vp):
    d = Diffuse(vp, EpsParametrization())
    assert d.is_preprocessor
    assert not d.is_postprocessor
    assert d.process is vp

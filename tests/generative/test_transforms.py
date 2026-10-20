import pytest
import torch

from schnetpack import properties
from schnetpack.generative import (
    VE,
    VP,
    Diffuse,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    Interpolant,
    PermutationCoupling,
    ScoreParametrization,
    Variable,
    VELinear,
    VelocityParametrization,
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


def positions(process, parametrization=None, name="positions"):
    return Variable(
        name=name,
        key=properties.R,
        per="atom",
        noising=process,
        parametrization=parametrization or EpsParametrization(),
        output_key="prediction",
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


@pytest.fixture
def vp():
    return GaussianInterpolant(VP())


# --- what it writes ------------------------------------------------------- #


def test_writes_x_t_label_weight_and_times(vp):
    torch.manual_seed(0)
    inputs = structure(7)
    x0 = inputs[properties.R].clone()

    out = Diffuse([positions(vp)])(inputs)

    assert out[properties.R].shape == x0.shape
    assert not torch.allclose(out[properties.R], x0)  # positions were noised
    assert out["label_positions"].shape == x0.shape
    assert torch.equal(out["weight_positions"], torch.ones(7))
    assert out["t"].shape == (7,)  # per atom, for conditioning
    assert out["t_structure"].shape == (1,)  # per structure, for a time head


def test_times_agree_across_granularities(vp):
    torch.manual_seed(0)
    out = Diffuse([positions(vp)])(structure(5))
    # one time per structure, broadcast — not five independent draws
    assert torch.allclose(out["t"], out["t_structure"].expand(5))
    assert out["t"].unique().numel() == 1


def test_time_lands_in_the_processs_usable_range(vp):
    torch.manual_seed(0)
    transform = Diffuse([positions(vp)])
    times = [transform(structure())["t_structure"].item() for _ in range(200)]
    assert min(times) >= vp.schedule.t_min
    assert max(times) <= vp.schedule.t_max
    assert min(times) < 0.2 and max(times) > 0.8  # covers the range


def test_keeps_the_clean_structure_when_asked(vp):
    torch.manual_seed(0)
    inputs = structure()
    x0 = inputs[properties.R].clone()
    out = Diffuse([positions(vp)], keep_clean=True)(inputs)
    assert torch.allclose(out["clean_positions"], x0)


def test_optional_keys_are_skipped_when_none(vp):
    out = Diffuse([positions(vp)], structure_time_key=None)(structure())
    assert "t_structure" not in out
    assert "clean_positions" not in out


def test_untouched_properties_survive(vp):
    inputs = structure(6)
    z = inputs[properties.Z].clone()
    out = Diffuse([positions(vp)])(inputs)
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
@pytest.mark.parametrize("process_cls", [VP, VE, VELinear], ids=lambda c: c.__name__)
def test_label_matches_the_parametrizations_target(process_cls, param_cls):
    # The whole point of "arbitrary label": swapping the parametrization must
    # swap the target with no other change.
    torch.manual_seed(0)
    schedule = VE(2e-4, 1.0) if process_cls is VE else process_cls()
    process = GaussianInterpolant(schedule)
    parametrization = param_cls()
    inputs = structure(6)
    x0 = inputs[properties.R].clone()

    out = Diffuse([positions(process, parametrization)])(inputs)

    # recover the noise the transform drew, then rebuild the target by hand
    t = out["t"]
    a = expand_t(process.schedule.a(t), x0)
    b = expand_t(process.schedule.b(t), x0)
    x1 = (out[properties.R] - a * x0) / b

    expected = parametrization.target(process.schedule, x0, x1, t)
    assert torch.allclose(out["label_positions"], expected, rtol=1e-4, atol=1e-5)


def test_eps_label_is_the_noise_that_made_x_t(vp):
    # The concrete case: label == (x_t - a x0)/b.
    torch.manual_seed(0)
    inputs = structure(6)
    x0 = inputs[properties.R].clone()
    out = Diffuse([positions(vp)])(inputs)

    t = out["t"]
    a = expand_t(vp.schedule.a(t), x0)
    b = expand_t(vp.schedule.b(t), x0)
    reconstructed = a * x0 + b * out["label_positions"]
    assert torch.allclose(reconstructed, out[properties.R], rtol=1e-5, atol=1e-6)


def test_x0_label_is_the_clean_structure(vp):
    torch.manual_seed(0)
    inputs = structure()
    x0 = inputs[properties.R].clone()
    out = Diffuse([positions(vp, X0Parametrization())])(inputs)
    assert torch.allclose(out["label_positions"], x0)


# --- the axes stay swappable ---------------------------------------------- #


@pytest.mark.parametrize(
    "schedule", [VP(), VE(2e-4, 1.0), VELinear()], ids=lambda s: type(s).__name__
)
def test_any_schedule_works(schedule):
    torch.manual_seed(0)
    out = Diffuse([positions(GaussianInterpolant(schedule))])(structure())
    assert torch.isfinite(out[properties.R]).all()
    assert torch.isfinite(out["label_positions"]).all()


def test_prior_decides_the_noise():
    # The extension point for constrained noise is the prior, not the
    # transform: a prior that projects out the mean must give a mean-free
    # label and a mean-free displacement, with no change to the transform.
    class MeanFreePrior(GaussianPrior):
        def __call__(self, batch):
            z = super().__call__(batch)[properties.R]
            return {**batch, properties.R: z - z.mean(0, keepdim=True)}

    torch.manual_seed(0)
    process = Interpolant(FlowMatching(), prior=MeanFreePrior())
    v = positions(process, VelocityParametrization())
    out = Diffuse([v], keep_clean=True)(structure(8))

    # flow-matching velocity target is x1 - x0, and x1 is mean-free
    x1 = out["label_positions"] + out["clean_positions"]
    assert x1.mean(0).abs().max() < 1e-6


def test_reconfigured_process_flows_through_the_transform():
    # A re-paired assembly diffuses just as the plain one does, as long as its
    # parametrization is valid; the coupling gets the atoms' groups.
    process = Interpolant(VE(2e-4, 1.0), coupling=PermutationCoupling())
    out = Diffuse([positions(process, VelocityParametrization())])(structure(8))
    assert torch.isfinite(out["label_positions"]).all()


def test_time_sampler_hook_is_used(vp):
    out = Diffuse(
        [positions(vp)],
        t_sampler=lambda n, device: torch.full((n,), 0.42, device=device),
    )(structure())
    assert out["t_structure"].item() == pytest.approx(0.42)
    assert torch.allclose(out["t"], torch.full((7,), 0.42))


def test_custom_keys(vp):
    out = Diffuse(
        [positions(vp, name="eps")],
        time_key="time",
        structure_time_key="time_mol",
    )(structure())
    assert "label_eps" in out and "time" in out and "time_mol" in out
    assert "t" not in out and "t_structure" not in out


def test_can_diffuse_a_property_other_than_positions():
    torch.manual_seed(0)
    inputs = structure()
    inputs["velocities"] = torch.randn(7, 3)
    r = inputs[properties.R].clone()
    v = Variable(
        name="velocities",
        key="velocities",
        per="atom",
        noising=GaussianInterpolant(VP(), prior=GaussianPrior(key="velocities")),
        parametrization=EpsParametrization(),
        output_key="prediction",
    )
    out = Diffuse([v])(inputs)
    assert not torch.allclose(out["velocities"], torch.zeros(7, 3))
    assert torch.equal(out[properties.R], r)  # positions untouched


# --- several variables ---------------------------------------------------- #


def test_per_structure_rows_and_one_time():
    inputs = {**structure(5), "cell_y": torch.randn(1, 6)}
    out = Diffuse([cell_like(), positions(GaussianInterpolant(VP()))])(inputs)
    assert out["cell_y"].shape == (1, 6)
    assert out["label_cell"].shape == (1, 6)
    assert out["weight_cell"].shape == (1,)
    assert out["t"].shape == (5,)
    assert out["t_structure"].shape == (1,)
    assert torch.all(out["t"] == out["t_structure"])


def test_noises_in_order_and_each_variable_sees_the_decoded_ones():
    shift = 100.0

    def move_atoms(batch):
        # stands in for the cell's decode: the atoms follow the noisy cell
        moved = batch[properties.R] + shift * batch["cell_y"].sum()
        return {**batch, properties.R: moved}

    clean = structure()
    inputs = {**clean, "cell_y": torch.randn(1, 6)}
    v_pos = positions(
        GaussianInterpolant(VE(0.01, 10.0), prior=GaussianPrior(10.0, centered=False))
    )
    out = Diffuse([cell_like(decode=move_atoms), v_pos], keep_clean=True)(inputs)

    expected = clean[properties.R] + shift * out["cell_y"].sum()
    assert torch.allclose(out["clean_positions"], expected)
    sigma = v_pos.noising.schedule.sigma(out["t"])[:, None]
    x0_seen = out[properties.R] - sigma * out["label_positions"]  # VE: a = 1
    assert torch.allclose(x0_seen, expected, atol=1e-4)


def test_writes_the_variables_weight(vp):
    calls = []

    def weight(schedule, t, x_t):
        calls.append((schedule, t, x_t))
        return 2.0 * torch.ones_like(t)

    v = positions(vp)
    v.weight = weight
    out = Diffuse([v])(structure())
    assert torch.equal(out["weight_positions"], 2.0 * torch.ones(7))
    ((schedule, t, x_t),) = calls
    assert schedule is vp.schedule
    assert torch.equal(x_t, out[properties.R])


def test_draws_times_on_the_shared_range():
    a = positions(Interpolant(FlowMatching(0.2, 0.9)), VelocityParametrization())
    transform = Diffuse([a, cell_like()])
    assert (transform.t_min, transform.t_max) == (0.2, 0.9)
    t = transform.sample_t(1000)
    assert t.min() >= 0.2 and t.max() <= 0.9


def test_refuses_duplicate_names_disjoint_ranges_and_no_variables(vp):
    with pytest.raises(ValueError, match="unique"):
        Diffuse([positions(vp), positions(vp)])
    a = positions(Interpolant(FlowMatching(0.0, 0.3)), VelocityParametrization())
    b = positions(
        Interpolant(FlowMatching(0.5, 1.0)), VelocityParametrization(), name="b"
    )
    with pytest.raises(ValueError, match="share no time range"):
        Diffuse([a, b])
    with pytest.raises(ValueError, match="at least one variable"):
        Diffuse([])


# --- shape and dtype ------------------------------------------------------ #


def test_time_matches_the_property_dtype(vp):
    inputs = structure()
    inputs[properties.R] = inputs[properties.R].double()
    out = Diffuse([positions(vp)])(inputs)
    assert out["t"].dtype == torch.float64
    assert out[properties.R].dtype == torch.float64


def test_is_a_preprocessor(vp):
    d = Diffuse([positions(vp)])
    assert d.is_preprocessor
    assert not d.is_postprocessor
    assert d.variables[0].noising is vp

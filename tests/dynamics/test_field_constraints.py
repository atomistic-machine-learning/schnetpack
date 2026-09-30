import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    DirectDenoising,
    EulerMaruyama,
    FieldConstraint,
    ForceFieldCalculator,
    HarmonicRestraint,
    Heun,
    Relaxer,
    Sampler,
)
from schnetpack.generative import (
    VE,
    VP,
    PseudoForceParametrization,
    ScoreParametrization,
    VelocityParametrization,
    expand_t,
)
from tests.dynamics.test_sampling import IDLE, ShapedPrior, analytic_score, batch_model


def restrained_batch(n_atoms, pairs, lengths, constants, seed=0):
    """A float64 batch of structures of ``n_atoms`` atoms with the given restraints."""
    generator = torch.Generator().manual_seed(seed)
    n_restraints = [len(p) for p in pairs]
    flat = [pair for structure in pairs for pair in structure]
    return {
        properties.n_atoms: torch.tensor(n_atoms),
        properties.idx_m: torch.repeat_interleave(
            torch.arange(len(n_atoms)), torch.tensor(n_atoms)
        ),
        properties.R: torch.randn(
            sum(n_atoms), 3, dtype=torch.float64, generator=generator
        ),
        HarmonicRestraint.n_restraints: torch.tensor(n_restraints),
        HarmonicRestraint.restraint_pairs: torch.tensor(flat, dtype=torch.long).view(
            -1, 2
        ),
        HarmonicRestraint.restraint_lengths: torch.tensor(lengths, dtype=torch.float64),
        HarmonicRestraint.restraint_constants: torch.tensor(
            constants, dtype=torch.float64
        ),
    }


def terms(batch, positions):
    """The restraint's energy and forces at ``positions``."""
    restraint = HarmonicRestraint()
    inputs = {**batch, properties.R: positions}
    return restraint.energy(inputs), restraint(inputs)


@pytest.fixture
def mixed_batch():
    # three structures of differing size, with 0, 1 and 2 restraints
    return restrained_batch(
        [2, 3, 4],
        [[], [(0, 2)], [(1, 3), (0, 1)]],
        [1.2, 0.8, 1.5],
        [10.0, 4.0, 2.0],
    )


# --- the restraint itself -------------------------------------------------- #


def test_harmonic_restraint_forces_are_the_negative_energy_gradient(mixed_batch):
    positions = mixed_batch[properties.R].clone().requires_grad_(True)
    energy, forces = terms(mixed_batch, positions)
    (grad,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.allclose(forces, -grad)


def test_harmonic_restraint_energy_per_structure(mixed_batch):
    positions = mixed_batch[properties.R]
    energy, forces = terms(mixed_batch, positions)

    def term(i, j, d0, k):
        return 0.5 * k * (torch.norm(positions[j] - positions[i]) - d0) ** 2

    # structure 1 starts at atom 2, structure 2 at atom 5
    expected = torch.stack(
        [
            torch.tensor(0.0, dtype=torch.float64),
            term(2, 4, 1.2, 10.0),
            term(6, 8, 0.8, 4.0) + term(5, 6, 1.5, 2.0),
        ]
    )
    assert torch.allclose(energy, expected)
    assert torch.equal(forces[:2], torch.zeros(2, 3, dtype=torch.float64))


def test_harmonic_restraint_checks_the_key_shapes(mixed_batch):
    batch = {**mixed_batch, HarmonicRestraint.restraint_lengths: torch.ones(2)}
    with pytest.raises(ValueError, match="restraint_lengths"):
        terms(batch, batch[properties.R])


def test_harmonic_restraint_returns_its_forces_and_leaves_the_batch(mixed_batch):
    before = dict(mixed_batch)
    out = HarmonicRestraint()(mixed_batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == mixed_batch[properties.R].shape
    assert mixed_batch.keys() == before.keys()
    for key, value in before.items():
        assert mixed_batch[key] is value


# --- relaxation: the restraint is part of the surface ---------------------- #


class ZeroModel:
    """A flat surface: whatever a relaxation does, the restraint did."""

    def __call__(self, inputs):
        positions = inputs[properties.R]
        n_structures = inputs[properties.n_atoms].shape[0]
        return {
            "energy": torch.zeros(n_structures, dtype=positions.dtype),
            "forces": torch.zeros_like(positions),
        }


def test_relaxer_relaxes_onto_the_restraint():
    batch = restrained_batch([3, 3], [[(0, 2)], [(0, 1)]], [1.3, 0.9], [5.0, 5.0])
    positions = Relaxer(ZeroModel(), constraints=[HarmonicRestraint()]).denoise(
        batch, 200, fmax=1e-4
    )[properties.R]
    assert torch.norm(positions[2] - positions[0]) == pytest.approx(1.3, abs=1e-4)
    assert torch.norm(positions[4] - positions[3]) == pytest.approx(0.9, abs=1e-4)
    # the unrestrained atom feels nothing and stays put
    assert torch.equal(positions[1], batch[properties.R][1])


def test_relaxer_holds_fixed_atoms_against_the_restraint():
    batch = restrained_batch([3], [[(0, 2)]], [1.3], [5.0])
    batch[properties.fixed_atoms] = torch.tensor([True, False, False])
    positions = Relaxer(ZeroModel(), constraints=[HarmonicRestraint()]).denoise(
        batch, 200, fmax=1e-4
    )[properties.R]
    assert torch.equal(positions[0], batch[properties.R][0])
    assert torch.norm(positions[2] - positions[0]) == pytest.approx(1.3, abs=1e-4)


def test_relaxer_evaluates_the_restraint_in_angstrom_whatever_the_models_units():
    # the model works in nm; the batch and the restraint's 1.3 Angstrom do not
    batch = restrained_batch([2], [[(0, 1)]], [1.3], [5.0])
    positions = Relaxer(
        ForceFieldCalculator(ZeroModel(), position_unit="nm"),
        constraints=[HarmonicRestraint()],
    ).denoise(batch, 200, fmax=1e-4)[properties.R]
    assert torch.norm(positions[1] - positions[0]) == pytest.approx(1.3, abs=1e-4)


def test_relaxer_sums_the_field_constraints():
    batch = restrained_batch([3], [[(0, 2)]], [1.3], [5.0])
    _, expected = terms(batch, batch[properties.R])
    relaxer = Relaxer(
        ZeroModel(), constraints=[HarmonicRestraint(), HarmonicRestraint()]
    )
    assert torch.allclose(relaxer._forces(batch), 2 * expected)


def test_relaxer_scales_each_field_constraint_by_its_weight():
    batch = restrained_batch([3], [[(0, 2)]], [1.3], [5.0])
    _, expected = terms(batch, batch[properties.R])
    relaxer = Relaxer(
        ZeroModel(),
        constraints=[HarmonicRestraint(weight=2.0), HarmonicRestraint(weight=0.5)],
    )
    assert torch.allclose(relaxer._forces(batch), 2.5 * expected)


# --- sampling: the restraint guides the score ------------------------------ #


def sampler_and_batch(parametrization, integrator, churn, weight, model):
    batch = restrained_batch([3, 2], [[(0, 2)], [(0, 1)]], [1.3, 0.9], [5.0, 5.0])
    sampler = Sampler(
        model,
        VP(),
        parametrization,
        integrator,
        churn=churn,
        constraints=[HarmonicRestraint(weight=weight)],
    )
    plain = Sampler(model, VP(), parametrization, integrator, churn=churn)
    t = torch.full((5,), 0.4, dtype=torch.float64)
    return sampler, plain, {**batch, properties.t: t}, t


def test_score_guidance_shifts_score_and_drift_by_the_weighted_forces():
    model = batch_model(analytic_score(VP(), 0.5, 1.0))
    sampler, plain, batch, t = sampler_and_batch(
        ScoreParametrization(), EulerMaruyama(), 1.0, 2.5, model
    )
    x = batch[properties.R]
    _, forces = terms(batch, x)

    guided, unguided = sampler.reverse(batch), plain.reverse(batch)
    assert torch.allclose(guided.score(x, t) - unguided.score(x, t), 2.5 * forces)
    g2 = expand_t(VP().sde().g2(t), x)
    assert torch.allclose(guided.drift(x, t) - unguided.drift(x, t), -g2 * 2.5 * forces)


def test_velocity_guidance_goes_through_g2():
    model = batch_model(lambda x, t: torch.zeros_like(x))
    sampler, plain, batch, t = sampler_and_batch(
        VelocityParametrization(), Heun(), 0.0, 2.5, model
    )
    x = batch[properties.R]
    _, forces = terms(batch, x)

    g2 = expand_t(VP().sde().g2(t), x)
    shift = sampler.reverse(batch).drift(x, t) - plain.reverse(batch).drift(x, t)
    assert torch.allclose(shift, -0.5 * g2 * 2.5 * forces)


def test_score_guidance_scales_each_field_constraint_by_its_weight():
    model = batch_model(analytic_score(VP(), 0.5, 1.0))
    sampler, plain, batch, t = sampler_and_batch(
        ScoreParametrization(), EulerMaruyama(), 1.0, 2.5, model
    )
    sampler.constraints = [HarmonicRestraint(weight=3.0), HarmonicRestraint(weight=0.5)]
    x = batch[properties.R]
    _, forces = terms(batch, x)
    shift = sampler.reverse(batch).score(x, t) - plain.reverse(batch).score(x, t)
    assert torch.allclose(shift, 3.5 * forces)


class TimeGuidance(FieldConstraint):
    """A guidance term that depends on the path time: t on every component."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, batch):
        t = batch[properties.t]
        self.seen.append(t)
        return t.unsqueeze(-1).expand_as(batch[properties.R]).clone()


def test_field_constraints_see_the_time_they_are_evaluated_at():
    model = batch_model(analytic_score(VP(), 0.5, 1.0))
    sampler, plain, batch, _ = sampler_and_batch(
        ScoreParametrization(), EulerMaruyama(), 1.0, 2.5, model
    )
    guidance = TimeGuidance()
    sampler.constraints = [guidance]
    x = batch[properties.R]
    # not the batch's own time: an integrator's evaluation point
    t = torch.full((5,), 0.7, dtype=torch.float64)
    shift = sampler.reverse(batch).score(x, t) - plain.reverse(batch).score(x, t)
    assert torch.equal(guidance.seen[-1], t)
    assert torch.allclose(shift, t.unsqueeze(-1).expand_as(x))


def test_velocity_guidance_needs_the_chart_at_assembly():
    process = VE(b_min=1e-2, prior=ShapedPrior())
    with pytest.raises(ValueError, match="chart"):
        Sampler(
            IDLE,
            process,
            VelocityParametrization(),
            Heun(),
            churn=0.0,
            constraints=[HarmonicRestraint()],
        )


def test_sampler_runs_with_a_restraint():
    model = batch_model(analytic_score(VP(), 0.0, 1.0))
    batch = restrained_batch([3, 2], [[(0, 2)], [(0, 1)]], [1.3, 0.9], [5.0, 5.0])
    sampler = Sampler(
        model,
        VP(),
        ScoreParametrization(),
        EulerMaruyama(),
        constraints=[HarmonicRestraint()],
    )
    out = sampler.denoise(batch, 10)
    assert torch.isfinite(out[properties.R]).all()


# --- who refuses what ------------------------------------------------------- #


def test_direct_denoising_refuses_field_constraints():
    with pytest.raises(ValueError, match="field constraint"):
        DirectDenoising(
            IDLE,
            VP(),
            PseudoForceParametrization(),
            constraints=[HarmonicRestraint()],
        )


def test_unknown_constraints_are_refused():
    with pytest.raises(TypeError, match="neither"):
        Relaxer(ZeroModel(), constraints=[object()])

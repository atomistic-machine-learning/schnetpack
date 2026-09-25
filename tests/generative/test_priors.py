import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import EulerMaruyama, Sampler
from schnetpack.generative import (
    VE,
    DatasetPrior,
    DatasetStructures,
    EpsParametrization,
    GaussianPrior,
    StatisticsStructures,
)


def positions(n, idx_m=None):
    """A batch whose positions give the draw its shape; with a layout if given."""
    batch = {properties.R: torch.empty(n, 3)}
    if idx_m is not None:
        batch[properties.idx_m] = idx_m
    return batch


# --- what the plain prior does, and why it is a problem -------------------- #


def test_an_uncentered_prior_is_off_the_zero_mean_subspace():
    # The defect the centered prior exists to fix: a fresh draw carries a mean
    # of scale std * sqrt(d / n) — here 10 * sqrt(3 / 12) = 5 A, larger than
    # the molecules the process is meant to generate.
    torch.manual_seed(0)
    prior = GaussianPrior(10.0, centered=False)
    draws = torch.stack(
        [prior.sample_positions(positions(12)).mean(0) for _ in range(200)]
    )
    assert draws.norm(dim=-1).mean().item() == pytest.approx(5.0, rel=0.2)


# --- centering ------------------------------------------------------------- #


def test_a_draw_without_segments_is_centered_as_one_group():
    torch.manual_seed(0)
    x = GaussianPrior(10.0).sample_positions(positions(12))
    assert x.mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_each_segment_is_centered_on_its_own():
    # The batch case: 8 molecules of 12 atoms concatenated along one axis.
    # Centering the batch as a whole would leave each molecule displaced, so
    # the per-segment means are what must vanish.
    torch.manual_seed(0)
    idx_m = torch.arange(8).repeat_interleave(12)
    x = GaussianPrior(10.0).sample_positions(positions(96, idx_m))
    for m in range(8):
        assert x[idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_a_batch_without_positions_takes_its_shape_from_the_atom_types():
    # A composition template: the positions are the prior's to draw.
    torch.manual_seed(0)
    idx_m = torch.arange(4).repeat_interleave(5)
    batch = {properties.idx_m: idx_m, properties.Z: torch.ones(20, dtype=torch.long)}
    x = GaussianPrior(3.0).sample_positions(batch)
    assert x.shape == (20, 3)
    for m in range(4):
        assert x[idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_a_mapping_without_idx_m_is_one_group():
    torch.manual_seed(0)
    x = GaussianPrior(3.0).sample_positions({properties.Z: torch.ones(20)})
    assert x.mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_segments_of_unequal_size_are_each_centered():
    torch.manual_seed(0)
    idx_m = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    x = GaussianPrior(5.0).sample_positions(positions(10, idx_m))
    for m in range(3):
        assert x[idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_a_mismatched_segment_count_is_refused():
    with pytest.raises(ValueError, match="one per row"):
        GaussianPrior(1.0).sample_positions(positions(10, torch.arange(4)))


def test_a_batch_without_positions_or_types_is_refused():
    with pytest.raises(KeyError, match="shape"):
        GaussianPrior(1.0).sample_positions({})


# --- still a Gaussian, on the subspace ------------------------------------- #


def test_the_scale_survives_centering_on_the_subspace():
    # Projection onto a subspace preserves the per-direction variance, so the
    # declared std stays the std the score conversions divide by. Measured in
    # the ambient basis the variance is std^2 (1 - 1/n) — the same statement.
    torch.manual_seed(0)
    n = 64
    x = GaussianPrior(2.0).sample_positions(positions(n))
    expected = 2.0 * (1.0 - 1.0 / n) ** 0.5
    assert x.std().item() == pytest.approx(expected, rel=0.1)


def test_it_declares_itself_gaussian_with_a_scale():
    # Both flags are load-bearing: gaussian gates the score/noise targets and
    # std gives sigma(t) = b(t) * std. A centered draw keeps both.
    prior = GaussianPrior(10.0)
    assert prior.gaussian is True
    assert prior.std == 10.0


def test_the_kernel_is_not_obstructed():
    assert VE(prior=GaussianPrior(10.0)).gaussian_kernel_obstruction() is None


def test_the_noise_parametrization_accepts_it():
    EpsParametrization().validate(VE(prior=GaussianPrior(10.0)))


# --- the two entry points -------------------------------------------------- #


def test_a_training_draw_is_shaped_like_the_positions_and_centered():
    # The training side: Diffuse runs per structure, so x0's whole leading
    # axis is one molecule and no layout is needed.
    torch.manual_seed(0)
    x0 = torch.randn(12, 3, dtype=torch.float64)
    x1 = GaussianPrior(10.0).sample_positions({properties.R: x0})
    assert x1.shape == x0.shape
    assert x1.dtype == x0.dtype
    assert x1.mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_perturb_stays_on_the_subspace_for_centered_data():
    # The precondition made visible: centered x0 plus centered x1 keeps x_t
    # centered at every time, which is what makes the targets learnable.
    torch.manual_seed(0)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))
    x0 = torch.randn(12, 3)
    x0 = x0 - x0.mean(0)
    for t in (0.1, 0.5, 1.0):
        x_t, _, x1, _, _ = process.perturb(x0, t=torch.full((12,), t))
        assert x_t.mean(0).norm().item() == pytest.approx(0.0, abs=1e-4)
        assert x1.mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_the_sampler_starts_a_batch_on_the_subspace():
    # The generation side: the sampler hands the batch to the prior, so the
    # start of a multi-molecule batch is centered per molecule.
    torch.manual_seed(0)
    idx_m = torch.arange(8).repeat_interleave(12)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))

    seen = {}

    def model(batch):
        x = batch[properties.R]
        seen.setdefault("x_init", x.clone())
        return {"prediction": torch.zeros_like(x)}

    sampler = Sampler(model, process, EpsParametrization(), EulerMaruyama())

    template = {properties.idx_m: idx_m, properties.R: torch.empty(96, 3)}
    sampler.denoise(sampler.prior.sample_from_batch(template), n_steps=2)
    for m in range(8):
        assert seen["x_init"][idx_m == m].mean(0).norm().item() == pytest.approx(
            0.0, abs=1e-4
        )


def test_given_positions_are_denoised_as_they_are():
    torch.manual_seed(0)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))
    x_init = torch.full((6, 3), 7.0)

    seen = {}

    def model(batch):
        x = batch[properties.R]
        seen.setdefault("x_init", x.clone())
        return {"prediction": torch.zeros_like(x)}

    sampler = Sampler(model, process, EpsParametrization(), EulerMaruyama())

    # the prior is not consulted, so the layout does not re-center them
    batch = {properties.R: x_init, properties.idx_m: torch.zeros(6, dtype=torch.long)}
    sampler.denoise(batch, n_steps=1)
    assert torch.allclose(seen["x_init"], x_init)


# --- the process derives its start from it, unchanged ---------------------- #


def test_the_sampling_prior_is_the_training_prior():
    prior = GaussianPrior(10.0)
    assert VE(prior=prior).sampling_prior() is prior


# --- the flag -------------------------------------------------------------- #


def test_centering_is_the_default():
    # Molecules are the point of this library, and for molecules the endpoint
    # belongs in the same zero-COM subspace as the data.
    assert GaussianPrior(1.0).centered is True


def test_centering_can_be_switched_off():
    # For data with no translation symmetry to quotient out, or a leading axis
    # of independent samples rather than the atoms of one structure.
    torch.manual_seed(0)
    x = GaussianPrior(10.0, centered=False).sample_positions(positions(12))
    assert x.mean(0).norm().item() > 1.0


def test_the_sugar_constructor_gets_a_centered_prior():
    # VE(sigma_min, sigma_max) builds its own GaussianPrior(sigma_max); the
    # default has to carry through, or the common path stays off-subspace.
    process = VE(sigma_min=0.05, sigma_max=10.0)
    assert process.prior.centered is True
    assert process.prior.std == 10.0


# --- the training side supplies the layout on its own ---------------------- #


def test_diffuse_centers_each_molecule_of_a_collated_batch():
    # Diffuse hands the batch to the prior as context, so a batch that carries
    # idx_m is centered per molecule and not as one cloud.
    from schnetpack.generative import Diffuse, EpsParametrization

    torch.manual_seed(0)
    idx_m = torch.arange(4).repeat_interleave(6)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))
    transform = Diffuse(process, EpsParametrization(), label_key="eps", time_key="t")

    x0 = GaussianPrior.center(torch.randn(24, 3), idx_m)  # centered data, per molecule
    out = transform(
        {
            properties.R: x0,
            properties.Z: torch.ones(24, dtype=torch.long),
            properties.idx_m: idx_m,
            properties.n_atoms: torch.full((4,), 6),
        }
    )
    for m in range(4):
        assert out["eps"][idx_m == m].mean(0).norm().item() == pytest.approx(
            0.0, abs=1e-4
        )
        assert out[properties.R][idx_m == m].mean(0).norm().item() == pytest.approx(
            0.0, abs=1e-4
        )


def test_diffuse_per_structure_centers_the_one_molecule():
    # The dataloader path: no idx_m before collation, and the whole leading
    # axis is the single structure being preprocessed.
    from schnetpack.generative import Diffuse, EpsParametrization

    torch.manual_seed(0)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))
    transform = Diffuse(process, EpsParametrization(), label_key="eps", time_key="t")

    x0 = torch.randn(9, 3)
    out = transform(
        {properties.R: x0 - x0.mean(0), properties.Z: torch.ones(9, dtype=torch.long)}
    )
    assert out["eps"].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)
    assert out[properties.R].mean(0).norm().item() == pytest.approx(0.0, abs=1e-4)


# --- where sampled structures come from ------------------------------------ #


def molecule(z):
    """One structure as a dataset item: atom types, positions, count."""
    z = torch.tensor(z)
    return {
        properties.Z: z,
        properties.R: torch.arange(3.0 * len(z)).reshape(-1, 3),
        properties.n_atoms: torch.tensor([len(z)]),
    }


DATASET = [molecule([1, 6]), molecule([8, 1, 1]), molecule([6, 6, 1, 1])]


def test_dataset_structures_collate_like_the_dataloader():
    batch = DatasetStructures(DATASET, shuffle=False).sample(2)
    assert torch.equal(batch[properties.Z], torch.tensor([1, 6, 8, 1, 1]))
    assert torch.equal(batch[properties.n_atoms], torch.tensor([2, 3]))
    assert torch.equal(batch[properties.idx_m], torch.tensor([0, 0, 1, 1, 1]))
    assert batch[properties.R].shape == (5, 3)


def test_dataset_structures_draw_every_structure_once_per_pass():
    torch.manual_seed(0)
    structures = DatasetStructures(DATASET)
    first = structures.sample(2)[properties.n_atoms].tolist()
    second = structures.sample(1)[properties.n_atoms].tolist()
    assert sorted(first + second) == [2, 3, 4]
    # the next pass starts over
    assert structures.sample(3)[properties.n_atoms].shape == (3,)


def test_dataset_structures_in_order_wrap_around():
    batch = DatasetStructures(DATASET, shuffle=False).sample(5)
    assert batch[properties.n_atoms].tolist() == [2, 3, 4, 2, 3]


def test_statistics_structures_follow_the_histograms():
    torch.manual_seed(0)
    structures = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 0, 1]),
        atom_types=torch.tensor([0, 0, 0, 0, 0, 0, 1]),
    )
    batch = structures.sample(4)
    assert batch[properties.n_atoms].tolist() == [3, 3, 3, 3]
    assert set(batch[properties.Z].tolist()) == {6}
    assert torch.equal(batch[properties.idx_m], torch.arange(4).repeat_interleave(3))
    assert properties.R not in batch


def test_statistics_structures_from_a_dataset():
    structures = StatisticsStructures.from_dataset(DATASET)
    assert structures.n_atoms.tolist() == [0, 0, 1, 1, 1]
    assert structures.atom_types[[1, 6, 8]].tolist() == [5, 3, 1]


def test_statistics_structures_refuse_empty_structures():
    with pytest.raises(ValueError, match="zero atoms"):
        StatisticsStructures(torch.tensor([1, 1]), torch.tensor([0, 1]))


def test_sample_draws_centered_positions_for_the_structures():
    torch.manual_seed(0)
    prior = GaussianPrior(10.0, structures=DatasetStructures(DATASET))
    batch = prior.sample(3)
    assert batch[properties.R].shape == (9, 3)
    for m in range(3):
        x = batch[properties.R][batch[properties.idx_m] == m]
        assert x.mean(0).norm().item() == pytest.approx(0.0, abs=1e-4)


def test_the_dataset_prior_returns_the_stored_structures():
    batch = DatasetPrior(DATASET, shuffle=False).sample(3)
    expected = torch.cat([m[properties.R] for m in DATASET])
    assert torch.equal(batch[properties.R], expected)


def test_the_dataset_prior_is_no_training_endpoint():
    with pytest.raises(TypeError, match="no positions law"):
        VE(prior=DatasetPrior(DATASET)).perturb(torch.randn(4, 3))

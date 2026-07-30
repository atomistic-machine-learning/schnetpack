import pytest
import torch

from schnetpack import properties
from schnetpack.generative import (
    VE,
    EpsParametrization,
    GaussianPrior,
    Sampler,
)
from schnetpack.generative.integrators import EulerMaruyama


# --- what the plain prior does, and why it is a problem -------------------- #


def test_an_uncentered_prior_is_off_the_zero_mean_subspace():
    # The defect the centered prior exists to fix: a fresh draw carries a mean
    # of scale std * sqrt(d / n) — here 10 * sqrt(3 / 12) = 5 A, larger than
    # the molecules the process is meant to generate.
    torch.manual_seed(0)
    prior = GaussianPrior(10.0, centered=False)
    draws = torch.stack([prior.sample((12, 3)).mean(0) for _ in range(200)])
    assert draws.norm(dim=-1).mean().item() == pytest.approx(5.0, rel=0.2)


# --- centering ------------------------------------------------------------- #


def test_a_draw_without_segments_is_centered_as_one_group():
    torch.manual_seed(0)
    x = GaussianPrior(10.0).sample((12, 3))
    assert x.mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_each_segment_is_centered_on_its_own():
    # The batch case: 8 molecules of 12 atoms concatenated along one axis.
    # Centering the batch as a whole would leave each molecule displaced, so
    # the per-segment means are what must vanish.
    torch.manual_seed(0)
    idx_m = torch.arange(8).repeat_interleave(12)
    x = GaussianPrior(10.0).sample((96, 3), context=idx_m)
    for m in range(8):
        assert x[idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_segments_come_from_a_batch_dict_too():
    # A SchNetPack batch can be handed over as-is; idx_m is read out of it.
    torch.manual_seed(0)
    idx_m = torch.arange(4).repeat_interleave(5)
    batch = {properties.idx_m: idx_m, properties.Z: torch.ones(20, dtype=torch.long)}
    x = GaussianPrior(3.0).sample((20, 3), context=batch)
    for m in range(4):
        assert x[idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_a_mapping_without_idx_m_is_one_group():
    torch.manual_seed(0)
    x = GaussianPrior(3.0).sample((20, 3), context={properties.Z: torch.ones(20)})
    assert x.mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_segments_of_unequal_size_are_each_centered():
    torch.manual_seed(0)
    idx_m = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    x = GaussianPrior(5.0).sample((10, 3), context=idx_m)
    for m in range(3):
        assert x[idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)


def test_a_mismatched_segment_count_is_refused():
    with pytest.raises(ValueError, match="one per row"):
        GaussianPrior(1.0).sample((10, 3), context=torch.arange(4))


def test_an_unusable_context_is_refused():
    with pytest.raises(TypeError, match="segment ids"):
        GaussianPrior(1.0).sample((10, 3), context="molecules")


# --- still a Gaussian, on the subspace ------------------------------------- #


def test_the_scale_survives_centering_on_the_subspace():
    # Projection onto a subspace preserves the per-direction variance, so the
    # declared std stays the std the score conversions divide by. Measured in
    # the ambient basis the variance is std^2 (1 - 1/n) — the same statement.
    torch.manual_seed(0)
    n = 64
    x = GaussianPrior(2.0).sample((n, 3))
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


def test_sample_like_centers_a_training_draw():
    # The training side: Diffuse runs per structure, so x0's whole leading
    # axis is one molecule and no context is needed.
    torch.manual_seed(0)
    x0 = torch.randn(12, 3)
    x1 = GaussianPrior(10.0).sample_like(x0)
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
    # The generation side: the sampler passes context to the prior, so the
    # start of a multi-molecule batch is centered per molecule.
    torch.manual_seed(0)
    idx_m = torch.arange(8).repeat_interleave(12)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))
    sampler = Sampler(process, EpsParametrization(), EulerMaruyama())

    seen = {}

    def model(x, t, cond):
        seen.setdefault("x_init", x.clone())
        return torch.zeros_like(x)

    sampler.sample(model, shape=(96, 3), n_steps=2, context={properties.idx_m: idx_m})
    for m in range(8):
        assert seen["x_init"][idx_m == m].mean(0).norm().item() == pytest.approx(
            0.0, abs=1e-4
        )


def test_an_explicit_x_init_ignores_the_context():
    torch.manual_seed(0)
    process = VE(b_min=0.05 / 10.0, prior=GaussianPrior(10.0))
    sampler = Sampler(process, EpsParametrization(), EulerMaruyama())
    x_init = torch.full((6, 3), 7.0)

    seen = {}

    def model(x, t, cond):
        seen.setdefault("x_init", x.clone())
        return torch.zeros_like(x)

    sampler.sample(model, shape=(6, 3), n_steps=1, x_init=x_init)
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
    x = GaussianPrior(10.0, centered=False).sample((12, 3))
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
        assert out["eps"][idx_m == m].mean(0).norm().item() == pytest.approx(0.0, abs=1e-4)
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
    out = transform({properties.R: x0 - x0.mean(0), properties.Z: torch.ones(9, dtype=torch.long)})
    assert out["eps"].mean(0).norm().item() == pytest.approx(0.0, abs=1e-5)
    assert out[properties.R].mean(0).norm().item() == pytest.approx(0.0, abs=1e-4)

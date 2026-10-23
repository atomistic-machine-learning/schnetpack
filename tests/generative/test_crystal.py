"""
Crystals: the cell as standardized parameters, read from the structure and
written back with the atoms following it, the cosine schedule and SNR weight
of the cell, and periodic positions with their wrapped-normal target.
"""

import math

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.cell import Cell
from scipy.spatial.transform import Rotation

from schnetpack import properties
from schnetpack.generative import (
    MASK,
    VE,
    VP,
    CellParameters,
    Diffuse,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    LogitsParametrization,
    MaskedTypeWeight,
    MaskNoising,
    PeriodicGaussianInterpolant,
    PeriodicPositions,
    SNRWeight,
    UniformCellPrior,
    Variable,
    VPCosine,
)
from schnetpack.utils.cell import (
    atom_cells,
    cell_statistics,
    cell_to_params,
    minimum_image,
    params_to_cell,
    to_cart,
    to_frac,
    wrap,
)

T = 1000

# MP-20 statistics of GEODE (Lou et al. 2026, App. D.1)
CELL = CellParameters(length=2.94, angle=99.7, var_length=2.52, var_angle=297.0)


def random_cells(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, 3, 3, generator=generator, dtype=torch.float64)
    return 4.0 * torch.eye(3, dtype=torch.float64) + noise


def rotated(cells, seed=0):
    """The cells turned by random rotations, out of any standard orientation."""
    rotations = Rotation.random(len(cells), random_state=seed).as_matrix()
    return cells @ torch.tensor(rotations, dtype=cells.dtype)


def crystal(atoms):
    """One structure as a dataset returns it."""
    return {
        properties.idx: torch.tensor([0]),
        properties.n_atoms: torch.tensor([len(atoms)]),
        properties.Z: torch.tensor(atoms.numbers),
        properties.R: torch.tensor(atoms.positions),
        properties.cell: torch.tensor(atoms.cell.array)[None],
        properties.pbc: torch.tensor(atoms.pbc),
    }


def batch_of(cells, n_atoms, seed=0):
    """Collated structures in ``cells`` with atoms at random fractional coordinates."""
    generator = torch.Generator().manual_seed(seed)
    n_atoms = torch.tensor(n_atoms)
    n = int(n_atoms.sum())
    batch = {
        properties.n_atoms: n_atoms,
        properties.idx_m: torch.repeat_interleave(torch.arange(len(n_atoms)), n_atoms),
        properties.Z: torch.ones(n, dtype=torch.long),
        properties.cell: cells,
        properties.pbc: torch.ones(3 * len(n_atoms), dtype=torch.bool),
    }
    frac = torch.rand(n, 3, generator=generator, dtype=torch.float64)
    batch[properties.R] = to_cart(frac, atom_cells(batch))
    return batch, frac


def skewed_si():
    """Diamond Si in a skewed, unreduced basis of its lattice."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_cell([[1, 0, 0], [1, 1, 0], [-1, 1, 1]] @ atoms.cell.array)
    atoms.wrap()
    return atoms


def periodic_distances(atoms_or_sample):
    """Sorted distances to all neighbors within 6 A, through the cell."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    if isinstance(atoms_or_sample, dict):
        s = atoms_or_sample
        atoms_or_sample = Atoms(
            numbers=s[properties.Z].numpy(),
            positions=s[properties.R].numpy(),
            cell=s[properties.cell].view(3, 3).numpy(),
            pbc=True,
        )
    return np.sort(neighbor_list("d", atoms_or_sample, 6.0))


def cell_variable(view=CELL):
    return Variable(
        name="cell",
        view=view,
        per="structure",
        noising=GaussianInterpolant(
            VPCosine(t_min=1.0 / T),
            # the batch holds the cell, never y: a key it does not hold
            prior=GaussianPrior(
                std=1.0, centered=False, key="cell_y", per="structure", dim=6
            ),
        ),
        parametrization=EpsParametrization(),
        output_key="eps_cell",
    )


def types_variable():
    return Variable(
        name="types",
        view=properties.Z,
        per="atom",
        noising=MaskNoising(FlowMatching(t_min=1.0 / T, t_max=1.0)),
        parametrization=LogitsParametrization(),
        weight=MaskedTypeWeight(n_steps=T),
        output_key="type_logits",
    )


def positions_variable(n_images=3):
    return Variable(
        name="positions",
        view=PeriodicPositions(),
        per="atom",
        noising=PeriodicGaussianInterpolant(
            VE(sigma_min=0.01, sigma_max=10.0), n_images=n_images
        ),
        parametrization=EpsParametrization(),
        output_key="eps_R",
    )


# --- the cell's parameters ------------------------------------------------------ #


def test_cell_parameters_are_standardized_per_cube_root_of_n():
    batch, _ = batch_of(random_cells(4), [1, 4, 8, 20])
    params = cell_to_params(batch[properties.cell])
    n_atoms = batch[properties.n_atoms]
    y = CELL.read(batch)
    torch.testing.assert_close(CELL.params(y, n_atoms), params)
    torch.testing.assert_close(
        CELL.mean(torch.tensor([8]), params),
        torch.tensor([[5.88] * 3 + [99.7] * 3], dtype=torch.float64),
    )
    std = CELL.std(params)
    assert math.isclose(std[0], math.sqrt(2.52 / 64)) and math.isclose(
        std[3], math.sqrt(297.0 / 64)
    )


def test_cell_statistics_standardize_the_cells_they_come_from():
    cells = rotated(random_cells(50))
    n_atoms = torch.randint(1, 21, (50,), generator=torch.Generator().manual_seed(0))
    view = CellParameters(**cell_statistics(cells, n_atoms), shrink=1.0)
    y = view.read({properties.cell: cells, properties.n_atoms: n_atoms})
    # the std of a length does not grow with N^(1/3), as in GEODE (App. D.1):
    # y of the lengths divided by N^(1/3) is standardized, pooled
    lengths = y[:, :3] / n_atoms.double()[:, None].pow(1.0 / 3.0)
    for block in (lengths, y[:, 3:]):
        assert abs(float(block.mean())) < 1e-10
        assert float(block.var()) == pytest.approx(1.0)


@pytest.mark.parametrize("n_atoms", [[3], [2, 3]])
def test_reading_a_written_cell_gives_its_parameters_back(n_atoms):
    torch.manual_seed(0)
    batch, _ = batch_of(rotated(random_cells(len(n_atoms))), n_atoms)
    y = torch.randn(len(n_atoms), 6, dtype=torch.float64)
    torch.testing.assert_close(CELL.read(CELL.write(batch, y)), y)


@pytest.mark.parametrize("handedness", [1.0, -1.0])
def test_writing_the_read_cell_keeps_the_structure(handedness):
    cells = rotated(random_cells(3))
    cells[:, 2] *= handedness  # c flipped: a left-handed cell
    batch, _ = batch_of(cells, [2, 3, 4])
    assert torch.all(torch.sign(torch.linalg.det(cells)) == handedness)
    out = CELL.write(batch, CELL.read(batch))
    assert set(out) == set(batch)
    for key in (properties.cell, properties.R):
        torch.testing.assert_close(out[key], batch[key])


def test_write_moves_the_atoms_with_the_cell_and_keeps_its_orientation():
    torch.manual_seed(0)
    cells = rotated(random_cells(2))
    cells[1, 2] *= -1.0
    batch, frac = batch_of(cells, [2, 3])
    y = torch.randn(2, 6, dtype=torch.float64)
    out = CELL.write(batch, y)
    new = out[properties.cell]
    torch.testing.assert_close(
        cell_to_params(new), CELL.params(y, batch[properties.n_atoms])
    )
    torch.testing.assert_close(to_frac(out[properties.R], atom_cells(out)), frac)
    # a along the old a, b in the old ab-plane, c on the same side of it
    a, b, c = new.unbind(1)
    a_old, b_old, c_old = cells.unbind(1)
    normal = torch.linalg.cross(a_old, b_old)
    torch.testing.assert_close(
        torch.nn.functional.cosine_similarity(a, a_old, dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        (b * normal).sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )
    assert torch.all(
        torch.sign((c * normal).sum(-1)) == torch.sign((c_old * normal).sum(-1))
    )


def test_write_takes_y_in_the_structures_dtype():
    batch, frac = batch_of(rotated(random_cells(2)), [2, 3])
    y = torch.randn(2, 6)  # float32, as a prior draws from the layout
    out = CELL.write(batch, y)
    assert out[properties.cell].dtype == torch.float64
    torch.testing.assert_close(CELL.read(out), y.double())
    torch.testing.assert_close(to_frac(out[properties.R], atom_cells(out)), frac)


def test_a_structure_without_a_cell_gets_the_standard_orientation():
    batch = {
        properties.n_atoms: torch.tensor([2, 3]),
        properties.idx_m: torch.tensor([0, 0, 1, 1, 1]),
        properties.cell: torch.zeros(2, 3, 3, dtype=torch.float64),
    }
    batch[properties.cell][1] = rotated(random_cells(1))[0]  # one has a cell
    y = torch.randn(2, 6, dtype=torch.float64)
    out = CELL.write(batch, y)
    params = CELL.params(y, batch[properties.n_atoms])
    torch.testing.assert_close(out[properties.cell][0], params_to_cell(params[0]))
    torch.testing.assert_close(CELL.read(out), y)
    assert properties.R not in out


@pytest.mark.parametrize(
    "params",
    [
        [4.0, 4.0, 4.0, 120.0, 120.0, 121.0],  # angles no cell has
        [4.0, 4.0, 4.0, 60.0, 60.0, 130.0],
        [4.0, -1.0, 4.0, 90.0, 90.0, 90.0],
        [4.0, 4.0, 4.0, 90.0, 190.0, 90.0],
    ],
)
def test_write_refuses_parameters_no_cell_has(params):
    batch, _ = batch_of(random_cells(2), [1, 1])
    n_atoms = batch[properties.n_atoms]
    params = torch.tensor(
        [[4.0, 4.0, 4.0, 90.0, 90.0, 90.0], params], dtype=torch.float64
    )
    y = (params - CELL.mean(n_atoms, params)) / CELL.std(params)
    with pytest.raises(ValueError, match="1 of 2 structures have cell parameters"):
        CELL.write(batch, y)


def test_periodic_positions_are_written_wrapped_into_the_cell():
    batch, frac = batch_of(rotated(random_cells(2)), [2, 3])
    view = PeriodicPositions()
    torch.testing.assert_close(
        view.read(view.write(batch, batch[properties.R])), batch[properties.R]
    )
    shifted = to_cart(
        frac + torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64), atom_cells(batch)
    )
    torch.testing.assert_close(
        view.write(batch, shifted)[properties.R], batch[properties.R]
    )


# --- cosine schedule and SNR weight ---------------------------------------------- #


def test_vp_cosine_runs_from_data_to_the_unit_prior():
    schedule = VPCosine()
    t = torch.tensor([0.0, 1e-3, 1.0], dtype=torch.float64)
    a, sigma = schedule.a(t), schedule.sigma(t)
    assert math.isclose(a[0], 1.0) and abs(a[2]) < 1e-15
    assert 1 - a[1] ** 2 == pytest.approx(4.1e-5, rel=0.01)
    torch.testing.assert_close(a**2 + sigma**2, torch.ones_like(t))
    assert schedule.sigma_max == 1.0


def test_vp_cosine_keeps_its_digits_at_small_t():
    schedule = VPCosine()
    t64 = torch.tensor([1e-3, 1e-2], dtype=torch.float64)
    torch.testing.assert_close(
        schedule.sigma(t64.float()), schedule.sigma(t64).float(), rtol=1e-6, atol=0
    )


def test_vp_cosine_derivatives_match_autograd():
    schedule = VPCosine()
    t = torch.tensor([0.01, 0.3, 0.7, 0.95], dtype=torch.float64)
    for analytic, fn in [
        (schedule.a_dot, schedule.a),
        (schedule.sigma_dot, schedule.sigma),
        (schedule.log_a_dot, lambda s: torch.log(schedule.a(s))),
    ]:
        t_ = t.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(fn(t_).sum(), t_)
        torch.testing.assert_close(analytic(t), grad)


def test_snr_weighted_x0_loss_is_the_loss_of_the_implied_noise():
    schedule = VPCosine()
    torch.manual_seed(0)
    t = torch.rand(7, dtype=torch.float64)
    y0, eps, y0_hat = torch.randn(3, 7, 6, dtype=torch.float64)
    a, sigma = schedule.a(t)[:, None], schedule.sigma(t)[:, None]
    y_t = a * y0 + sigma * eps
    eps_hat = (y_t - a * y0_hat) / sigma
    weight = SNRWeight()(schedule, t, y_t)
    torch.testing.assert_close(
        weight * ((y0_hat - y0) ** 2).sum(1), ((eps_hat - eps) ** 2).sum(1)
    )


# --- periodic positions ---------------------------------------------------------- #


def test_periodic_positions_need_a_variance_exploding_schedule():
    with pytest.raises(ValueError, match="variance-exploding"):
        PeriodicGaussianInterpolant(VP())


def test_periodic_positions_are_noised_in_the_cell_and_wrapped():
    torch.manual_seed(0)
    sample = crystal(skewed_si())
    noising = PeriodicGaussianInterpolant(VE(sigma_min=0.01, sigma_max=10.0))
    t = torch.full((2,), 0.8, dtype=torch.float64)
    x_t, x0, x1, _ = noising.perturb(sample[properties.R], t, batch=sample)
    frac = to_frac(x_t, sample[properties.cell][0])
    assert frac.min() >= 0 and frac.max() < 1
    assert x1.shape == x0.shape


def brute_force_eps(x_t, x0, cell, sigma, n=8):
    """The wrapped-normal eps target, summed over (2n + 1)^3 images in a loop."""
    targets = []
    for xt, xo in zip(x_t, x0):
        numerator, denominator = torch.zeros(3, dtype=torch.float64), 0.0
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    d = xt - xo + torch.tensor([i, j, k], dtype=torch.float64) @ cell
                    w = math.exp(-float(d @ d) / (2 * sigma**2))
                    numerator += w * d
                    denominator += w
        targets.append(numerator / denominator / sigma)
    return torch.stack(targets)


@pytest.mark.parametrize(
    "sigma, n_images, tolerance", [(0.5, 2, 1e-10), (3.0, 4, 1e-3)]
)
def test_wrapped_eps_target_matches_a_brute_force_image_sum(sigma, n_images, tolerance):
    torch.manual_seed(0)
    cell = torch.tensor(Cell.fromcellpar([4.0, 4.5, 5.0, 80, 100, 110]).array)
    schedule = VE(sigma_min=0.01, sigma_max=10.0)
    noising = PeriodicGaussianInterpolant(schedule, n_images=n_images)
    t = schedule.t_of_sigma(torch.tensor(sigma, dtype=torch.float64))
    x0 = torch.rand(6, 3, dtype=torch.float64) @ cell
    x_t = wrap(x0 + sigma * torch.randn(6, 3, dtype=torch.float64), cell)
    x1 = noising.effective_x1(x_t, x0, cell.expand(6, 3, 3), t)
    eps = EpsParametrization().target(schedule, x0, x1, t)
    torch.testing.assert_close(
        eps, brute_force_eps(x_t, x0, cell, sigma), atol=tolerance, rtol=0
    )


def test_wrapped_eps_target_is_the_noise_while_it_is_small():
    torch.manual_seed(0)
    cell = torch.tensor(Cell.fromcellpar([4.0, 4.5, 5.0, 80, 100, 110]).array)
    schedule = VE(sigma_min=0.01, sigma_max=10.0)
    noising = PeriodicGaussianInterpolant(schedule)
    sigma = 0.05
    t = schedule.t_of_sigma(torch.tensor(sigma, dtype=torch.float64))
    x0 = torch.rand(20, 3, dtype=torch.float64) @ cell
    eps = torch.randn(20, 3, dtype=torch.float64)
    x_t = wrap(x0 + sigma * eps, cell)
    x1 = noising.effective_x1(x_t, x0, cell.expand(20, 3, 3), t)
    torch.testing.assert_close(x1 / schedule.sigma_max, eps)
    # and x0 in another image gives the same target
    shifted = x0 + torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64) @ cell
    torch.testing.assert_close(
        noising.effective_x1(x_t, shifted, cell.expand(20, 3, 3), t), x1
    )


def test_uniform_cell_prior_fills_each_cell():
    torch.manual_seed(0)
    cells = random_cells(2)
    batch = {
        properties.n_atoms: torch.tensor([4000, 4000]),
        properties.idx_m: torch.repeat_interleave(torch.arange(2), 4000),
        properties.cell: cells,
    }
    r = UniformCellPrior()(batch)[properties.R]
    frac = to_frac(r, atom_cells(batch))
    assert frac.min() >= 0 and frac.max() < 1
    torch.testing.assert_close(
        frac.mean(0), torch.full((3,), 0.5, dtype=torch.float64), atol=0.01, rtol=0
    )


# --- Diffuse on a crystal ---------------------------------------------------------- #


def diffuse(t, n_images=3):
    return Diffuse(
        [cell_variable(), types_variable(), positions_variable(n_images)],
        t_sampler=lambda n, device: torch.full((n,), t, device=device),
    )


def test_diffuse_noises_the_cell_first_and_the_positions_in_it():
    torch.manual_seed(0)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.set_cell(
        rotated(torch.tensor(atoms.cell.array)[None])[0].numpy(), scale_atoms=True
    )
    sample = crystal(atoms)
    cell_0 = sample[properties.cell][0].clone()
    frac_0 = to_frac(sample[properties.R], cell_0)
    y0 = CELL.read(sample)
    out = diffuse(1e-3)(dict(sample))
    assert "cell_y" not in out  # the batch holds the cell only
    # the cell moved in its own orientation, the atoms kept their fractional
    # coordinates up to sigma
    cell_t = out[properties.cell][0]
    torch.testing.assert_close(
        torch.nn.functional.cosine_similarity(cell_t[0], cell_0[0], dim=0),
        torch.tensor(1.0, dtype=torch.float64),
    )
    frac_t = to_frac(out[properties.R], cell_t)
    gap = minimum_image(to_cart(frac_t - frac_0, cell_t), cell_t)
    assert gap.norm(dim=-1).max() < 0.1
    # the label is the noise that took y0 to the y of the noisy cell
    schedule = VPCosine(t_min=1.0 / T)
    tt = torch.tensor([1e-3], dtype=torch.float64)
    a, sigma = schedule.a(tt), schedule.sigma(tt)
    torch.testing.assert_close(out["label_cell"], (CELL.read(out) - a * y0) / sigma)
    torch.testing.assert_close(out["weight_cell"], torch.ones(1, dtype=torch.float64))
    assert out["t_structure"].shape == (1,) and out[properties.t].shape == (8,)


@pytest.mark.parametrize(
    "cellpar", [[3.0, 3.0, 3.0, 60, 60, 60], [5, 6, 7, 70, 95, 110]]
)
def test_position_noise_is_sigma_in_angstrom_whatever_the_cell(cellpar):
    torch.manual_seed(0)
    schedule = VE(sigma_min=0.01, sigma_max=10.0)
    noising = PeriodicGaussianInterpolant(schedule)
    cell = torch.tensor(Cell.fromcellpar(cellpar).array)
    n = 20000
    batch = {
        properties.Z: torch.ones(n, dtype=torch.long),
        properties.cell: cell[None],
    }
    x0 = torch.rand(n, 3, dtype=torch.float64) @ cell
    sigma = 0.2
    t = schedule.t_of_sigma(torch.tensor(sigma, dtype=torch.float64))
    x_t, *_ = noising.perturb(x0, t.expand(n), batch=batch)
    d = minimum_image(x_t - x0, cell)
    assert d.std().item() == pytest.approx(sigma, rel=0.02)


def test_at_t_one_the_crystal_is_the_sampling_start():
    torch.manual_seed(0)
    sample = crystal(bulk("Si", "diamond", a=5.43, cubic=True))
    ys, fracs = [], []
    for _ in range(400):
        out = diffuse(1.0, n_images=1)({k: v.clone() for k, v in sample.items()})
        assert (out[properties.Z] == MASK).all()
        ys.append(CELL.read(out))
        fracs.append(to_frac(out[properties.R], out[properties.cell][0]))
    ys, fracs = torch.cat(ys), torch.cat(fracs)
    # the cell from its unit prior, the positions uniform in it
    assert ys.mean().abs() < 0.1 and (ys.std() - 1.0).abs() < 0.05
    torch.testing.assert_close(
        fracs.mean(0), torch.full((3,), 0.5, dtype=torch.float64), atol=0.02, rtol=0
    )
    assert (fracs.var(0) - 1.0 / 12.0).abs().max() < 0.01


def test_view_input_writes_the_cell_parameters_for_the_network():
    from schnetpack.generative import ViewInput

    batch, _ = batch_of(rotated(random_cells(2)), [2, 3])
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    out = ViewInput(CELL, "cell_y")(dict(batch))
    torch.testing.assert_close(out["cell_y"], CELL.read(batch))
    assert out["cell_y"].dtype == torch.float32

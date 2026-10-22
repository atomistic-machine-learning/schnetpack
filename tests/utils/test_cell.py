"""Periodic geometry: cell parameters, the Niggli cell, wrapping, new cells."""

import numpy as np
import pytest
import torch
from ase.cell import Cell

from schnetpack import properties
from schnetpack.utils.cell import (
    atom_cells,
    cell_statistics,
    cell_to_params,
    minimum_image,
    niggli_reduce,
    params_to_cell,
    to_cart,
    to_frac,
    valid_params,
    with_cell,
    wrap,
)


def random_cells(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, 3, 3, generator=generator, dtype=torch.float64)
    return 4.0 * torch.eye(3, dtype=torch.float64) + noise


def test_params_to_cell_keeps_the_metric_of_a_cell():
    cells = random_cells(50)
    rebuilt = params_to_cell(cell_to_params(cells))
    torch.testing.assert_close(
        rebuilt @ rebuilt.mT, cells @ cells.mT, atol=1e-10, rtol=0
    )


def test_params_to_cell_is_in_standard_orientation():
    cell = params_to_cell(cell_to_params(random_cells(10)))
    assert torch.all(cell[:, 0, 1:] == 0) and torch.all(cell[:, 1, 2] == 0)
    assert torch.all(torch.linalg.det(cell) > 0)


def test_cell_to_params_match_ase():
    cells = random_cells(5)
    expected = np.stack([Cell(c.numpy()).cellpar() for c in cells])
    np.testing.assert_allclose(cell_to_params(cells).numpy(), expected, atol=1e-8)


def test_niggli_reduce_is_a_basis_of_the_same_lattice_and_reduced():
    for cell in random_cells(10, seed=3):
        reduced = niggli_reduce(cell)
        # integer change of basis with determinant 1
        change = reduced @ torch.linalg.inv(cell)
        torch.testing.assert_close(change, change.round(), atol=1e-8, rtol=0)
        assert round(float(torch.linalg.det(change))) == 1
        np.testing.assert_allclose(
            cell_to_params(reduced).numpy(),
            Cell(cell.numpy()).niggli_reduce()[0].cellpar(),
            atol=1e-8,
        )
        torch.testing.assert_close(niggli_reduce(reduced), reduced)


def test_wrap_and_minimum_image():
    torch.manual_seed(0)
    cells = random_cells(1)[0].expand(100, 3, 3)
    r = 20.0 * torch.randn(100, 3, dtype=torch.float64)
    wrapped = wrap(r, cells)
    frac = to_frac(wrapped, cells)
    assert frac.min() >= 0 and frac.max() < 1
    shift = to_frac(r - wrapped, cells)
    torch.testing.assert_close(shift, shift.round(), atol=1e-9, rtol=0)
    d = minimum_image(r, cells)
    assert to_frac(d, cells).abs().max() <= 0.5 + 1e-12


def test_parameters_of_cells_are_valid_and_impossible_ones_are_not():
    assert valid_params(cell_to_params(random_cells(50))).all()
    impossible = torch.tensor(
        [
            [4.0, 4.0, 4.0, 120.0, 120.0, 120.5],  # angles summing past 360
            [4.0, 4.0, 4.0, 60.0, 60.0, 130.0],  # one larger than the others' sum
            [4.0, 4.0, 0.0, 90.0, 90.0, 90.0],
            [4.0, 4.0, 4.0, 0.0, 90.0, 90.0],
            [4.0, 4.0, 4.0, 90.0, 180.0, 90.0],
            [4.0, 4.0, float("nan"), 90.0, 90.0, 90.0],
        ]
    )
    assert not valid_params(impossible).any()


@pytest.mark.parametrize("collated", [False, True])
def test_with_cell_keeps_the_fractional_coordinates(collated):
    torch.manual_seed(0)
    cells = random_cells(2 if collated else 1)
    frac = torch.rand(5, 3, dtype=torch.float64)
    inputs = {properties.cell: cells, properties.Z: torch.ones(5, dtype=torch.long)}
    if collated:
        inputs[properties.idx_m] = torch.tensor([0, 0, 1, 1, 1])
    inputs[properties.R] = to_cart(frac, atom_cells(inputs))
    new = random_cells(len(cells), seed=1)
    out = with_cell(inputs, new)
    assert out[properties.cell] is new
    torch.testing.assert_close(to_frac(out[properties.R], atom_cells(out)), frac)
    assert torch.equal(inputs[properties.cell], cells)  # a copy
    without_positions = with_cell({properties.cell: cells}, new)
    assert set(without_positions) == {properties.cell}


def test_cell_statistics_pool_lengths_per_cube_root_of_n_and_angles():
    cells = torch.stack(
        [torch.diag(torch.tensor([1.0, 2.0, 3.0])), 2.0 * torch.eye(3) * 8 ** (1 / 3)]
    )
    stats = cell_statistics(cells, torch.tensor([1, 8]))
    # lengths / N^(1/3): 1, 2, 3 and 2, 2, 2
    assert stats["length"] == pytest.approx(2.0)
    assert stats["var_length"] == pytest.approx(0.4)
    assert stats["angle"] == pytest.approx(90.0)
    assert stats["var_angle"] == pytest.approx(0.0, abs=1e-20)

"""Cell transforms: the Niggli cell and wrapping, on a structure and a batch."""

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.neighborlist import neighbor_list

from schnetpack import properties
from schnetpack.transform import NiggliReduce, WrapPositions
from schnetpack.utils.cell import (
    atom_cells,
    cell_to_params,
    niggli_reduce,
    to_cart,
    to_frac,
)


def structure(atoms):
    """One structure as a dataset returns it."""
    return {
        properties.n_atoms: torch.tensor([len(atoms)]),
        properties.Z: torch.tensor(atoms.numbers),
        properties.R: torch.tensor(atoms.positions),
        properties.cell: torch.tensor(atoms.cell.array)[None],
        properties.pbc: torch.tensor(atoms.pbc),
    }


def skewed_si():
    """Diamond Si in a skewed, unreduced basis of its lattice."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_cell([[1, 0, 0], [1, 1, 0], [-1, 1, 1]] @ atoms.cell.array)
    atoms.wrap()
    return atoms


def test_niggli_reduce_keeps_the_crystal():
    atoms = skewed_si()
    out = NiggliReduce()(structure(atoms))
    cell = out[properties.cell]
    assert cell.shape == (1, 3, 3)
    torch.testing.assert_close(cell[0], niggli_reduce(torch.tensor(atoms.cell.array)))
    assert not np.allclose(cell_to_params(cell).numpy(), atoms.cell.cellpar())
    torch.testing.assert_close(out[properties.R], torch.tensor(atoms.positions))
    reduced = atoms.copy()
    reduced.set_cell(cell[0].numpy())
    np.testing.assert_allclose(
        np.sort(neighbor_list("d", reduced, 6.0)),
        np.sort(neighbor_list("d", atoms, 6.0)),
        atol=1e-8,
    )


@pytest.mark.parametrize("collated", [False, True])
def test_wrap_positions_wraps_the_periodic_directions_only(collated):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    if collated:
        n_atoms = torch.tensor([2, 3])
        pbc = torch.tensor([True, True, True, True, False, True])
    else:
        n_atoms = torch.tensor([5])
        pbc = torch.tensor([True, False, True])
    n_structures, n = len(n_atoms), int(n_atoms.sum())
    cells = 4.0 * torch.eye(3, dtype=torch.float64) + torch.randn(
        n_structures, 3, 3, generator=generator, dtype=torch.float64
    )
    batch = {
        properties.n_atoms: n_atoms,
        properties.Z: torch.ones(n, dtype=torch.long),
        properties.cell: cells,
        properties.pbc: pbc,
    }
    if collated:
        batch[properties.idx_m] = torch.repeat_interleave(
            torch.arange(n_structures), n_atoms
        )
    frac = 3.0 * torch.randn(n, 3, generator=generator, dtype=torch.float64)
    batch[properties.R] = to_cart(frac, atom_cells(batch))
    out = WrapPositions()(dict(batch))
    new = to_frac(out[properties.R], atom_cells(batch))
    periodic = pbc.view(-1, 3)[
        batch.get(properties.idx_m, torch.zeros(n, dtype=torch.long))
    ]
    assert new[periodic].min() >= -1e-6 and new[periodic].max() < 1
    shift = new - frac
    torch.testing.assert_close(shift, shift.round(), atol=1e-9, rtol=0)
    torch.testing.assert_close(new[~periodic], frac[~periodic])

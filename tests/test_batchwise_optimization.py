import os

import numpy as np
import pytest
from ase.io import read

import schnetpack as spk
from schnetpack.interfaces.batchwise_optimization import (
    ASEBatchwiseLBFGS,
    BatchwiseCalculator,
)

TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")


@pytest.fixture
def calculator():
    # md_ethanol.model was saved with an older version and needs load_model
    return BatchwiseCalculator(
        os.path.join(TESTDATA, "md_ethanol.model"),
        spk.interfaces.AtomsConverter(
            neighbor_list=spk.transform.ASENeighborList(cutoff=5.0)
        ),
        energy_unit="kcal/mol",
        position_unit="Ang",
    )


def rattled_ethanol(n_structures):
    ethanol = read(os.path.join(TESTDATA, "md_ethanol.xyz"))
    structures = []
    for seed in range(n_structures):
        atoms = ethanol.copy()
        atoms.rattle(0.05, seed=seed)
        structures.append(atoms)
    return structures


def test_lbfgs_relaxes_every_structure_of_the_batch(calculator, tmp_path):
    trajectory = str(tmp_path / "relax")
    optimizer = ASEBatchwiseLBFGS(
        calculator, rattled_ethanol(3), logfile=None, trajectory=trajectory
    )

    # without a step limit, the optimizer runs until it has converged
    assert optimizer.run(fmax=0.05)
    assert optimizer.nsteps > 0

    forces = calculator.get_forces(optimizer.atoms)
    assert np.linalg.norm(forces, axis=1).max() < 0.05
    for i, atoms in enumerate(optimizer.atoms):
        last_frame = read(f"{trajectory}_{i}.xyz")
        np.testing.assert_allclose(last_frame.positions, atoms.positions, atol=1e-6)


def test_lbfgs_keeps_fixed_atoms_in_place(calculator):
    structures = rattled_ethanol(2)
    start = [atoms.positions.copy() for atoms in structures]
    # fix the first atom of every structure
    fixed = [i == 0 for i in range(len(structures[0]))] * len(structures)

    optimizer = ASEBatchwiseLBFGS(
        calculator, structures, logfile=None, fixed_atoms_mask=fixed
    )
    optimizer.run(fmax=0.05, steps=200)

    for atoms, positions in zip(optimizer.atoms, start):
        np.testing.assert_allclose(atoms.positions[0], positions[0])
        assert not np.allclose(atoms.positions[1:], positions[1:])

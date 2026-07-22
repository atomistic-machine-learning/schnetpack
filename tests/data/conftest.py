import os

import numpy as np
import pytest
from ase import Atoms

from schnetpack.data import ASEAtomsData

ENERGY = "energy"
H_ATOMREF = -2.5
O_ATOMREF = 4.5


def _build_stats_db(datapath, atomrefs=None):
    db = ASEAtomsData.create(
        datapath,
        distance_unit="Ang",
        property_unit_dict={ENERGY: "eV"},
        atomrefs=atomrefs,
    )

    rng = np.random.RandomState(42)
    atoms_list = []
    property_list = []
    for i in range(20):
        n_h = 1 + i % 4
        n_o = 1 + (i * 3) % 5
        numbers = [1] * n_h + [8] * n_o
        atoms_list.append(
            Atoms(numbers=numbers, positions=rng.randn(len(numbers), 3))
        )
        energy = -2.0 * n_h + 5.0 * n_o + 0.1 * (i % 7)
        property_list.append({ENERGY: np.array([energy])})

    db.add_systems(property_list, atoms_list)
    return datapath


@pytest.fixture
def stats_dbpath(tmp_path):
    """Small deterministic H/O dataset with known energies, no atomrefs."""
    return _build_stats_db(os.path.join(tmp_path, "stats_test.db"))


@pytest.fixture
def stats_dbpath_with_atomrefs(tmp_path):
    """Same dataset, with single-atom references in the DB metadata."""
    atomrefs = [0.0] * 100
    atomrefs[1] = H_ATOMREF
    atomrefs[8] = O_ATOMREF
    return _build_stats_db(
        os.path.join(tmp_path, "stats_test_atomrefs.db"),
        atomrefs={ENERGY: atomrefs},
    )

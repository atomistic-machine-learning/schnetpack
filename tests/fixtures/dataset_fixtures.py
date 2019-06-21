import pytest
import os
import torch
from ase import Atoms
from ase.db import connect
from src.schnetpack.data import AtomsData
from src.schnetpack.datasets import QM9
import schnetpack as spk
import numpy as np


__all__ = [
    "tmp_db_path",
    "db_size",
    "sub_ids",
    "dataset",
    "n_atoms",
    "ats",
    "build_db",
    "qm9_dbpath",
    "qm9_dataset",
    "qm9_split",
    "qm9_splits",
    "qm9_avlailable_properties",
]


@pytest.fixture(scope="session")
def tmp_db_path(tmpdir_factory):
    return os.path.join(tmpdir_factory.mktemp("data"), "test2.db")


@pytest.fixture(scope="session")
def db_size():
    return 20


@pytest.fixture(scope="session")
def sub_ids(db_size):
    return [1, 3, 5, 4, 10, 16]


@pytest.fixture(scope="session")
def dataset(tmp_db_path, build_db):
    return AtomsData(tmp_db_path, center_positions=False)


@pytest.fixture(scope="session")
def n_atoms():
    return 3


@pytest.fixture(scope="session")
def ats(db_size):
    n_atoms = [3, 5]
    n_small = np.random.randint(1, db_size - 1)

    molecules = []
    data = []
    for i in range(db_size):
        mol_type = int(i <= n_small)
        data.append(
            {
                "prop": np.random.rand(1),
                "der": np.random.rand(n_atoms[mol_type], 3),
                "contrib": np.random.rand(n_atoms[mol_type], 1),
            }
        )
        molecules.append(
            Atoms("N{}".format(n_atoms[mol_type]), np.random.rand(n_atoms[mol_type], 3))
        )
    return molecules, data


@pytest.fixture(scope="session")
def build_db(tmp_db_path, ats):
    molecules, data = ats
    with connect(tmp_db_path) as conn:
        for mol, properties in zip(molecules, data):
            conn.write(mol, data=properties)


@pytest.fixture(scope="module")
def qm9_dbpath():
    return os.path.join("tests", "data", "test_qm9.db")


@pytest.fixture(scope="module")
def qm9_dataset(qm9_dbpath):
    print(os.path.exists(qm9_dbpath))
    return QM9(qm9_dbpath)


@pytest.fixture(scope="module")
def qm9_split():
    return 10, 5


@pytest.fixture(scope="module")
def qm9_splits(qm9_dataset, qm9_split):
    return spk.data.train_test_split(qm9_dataset, *qm9_split)


@pytest.fixture(scope="session")
def qm9_avlailable_properties():
    return [
        "rotational_constant_A",
        "rotational_constant_B",
        "rotational_constant_C",
        "dipole_moment",
        "isotropic_polarizability",
        "homo",
        "lumo",
        "gap",
        "electronic_spatial_extent",
        "zpve",
        "energy_U0",
        "energy_U",
        "enthalpy_H",
        "free_energy",
        "heat_capacity",
    ]

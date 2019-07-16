import os
import pytest
import schnetpack as spk
from schnetpack import AtomsLoader
from schnetpack.datasets import QM9


__all__ = [
    "qm9_dbpath",
    "qm9_dataset",
    "qm9_split",
    "qm9_splits",
    "qm9_avlailable_properties",
    "qm9_train_loader",
    "qm9_val_loader",
    "qm9_test_loader",
]


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


@pytest.fixture(scope="module")
def qm9_train_loader(qm9_splits, batch_size, shuffle):
    return AtomsLoader(qm9_splits[0], batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="module")
def qm9_val_loader(qm9_splits, batch_size, shuffle):
    return AtomsLoader(qm9_splits[1], batch_size=batch_size, shuffle=shuffle)


@pytest.fixture(scope="module")
def qm9_test_loader(qm9_splits, batch_size, shuffle):
    return AtomsLoader(qm9_splits[2], batch_size=batch_size, shuffle=shuffle)

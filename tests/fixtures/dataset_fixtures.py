import pytest
import os
from ase import Atoms
from ase.db import connect
from src.schnetpack.data import AtomsData
from src.schnetpack.datasets import QM9
import schnetpack as spk


@pytest.fixture(scope="session")
def tmp_path(tmpdir_factory):
    return os.path.join(tmpdir_factory.mktemp("data"), "test2.db")


@pytest.fixture(scope="session")
def db_size():
    return 20


@pytest.fixture(scope="session")
def sub_ids(db_size):
    return [1, 3, 5, 4, 10, 16]


@pytest.fixture(scope="session")
def dataset(tmp_path, build_db):
    return AtomsData(tmp_path, center_positions=False)


@pytest.fixture(scope="session")
def n_atoms():
    return 3


@pytest.fixture(scope="session")
def ats():
    return Atoms("N3", [(0, 0, 0), (1, 0, 0), (0, 0, 1)])


@pytest.fixture(scope="session")
def build_db(tmp_path, db_size, ats):
    with connect(tmp_path) as conn:
        for mol in [ats] * db_size:
            conn.write(mol)


@pytest.fixture(scope="module")
def qm9_dbpath():
    return os.path.join("data", "test_qm9.db")


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

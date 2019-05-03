import pytest
import os
from ase import Atoms
from ase.db import connect
from src.schnetpack.data import AtomsData


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

import os
import pytest
import numpy as np
import schnetpack as spk
from ase import Atoms
from ase.db import connect


@pytest.fixture(scope="session")
def db_size():
    return 20


@pytest.fixture(scope="session")
def tmp_db_path(tmpdir_factory):
    return os.path.join(tmpdir_factory.mktemp("data"), "test2.db")


@pytest.fixture(scope="session")
def db_config(db_size):
    n_small = np.random.randint(1, db_size - 1)
    return [[n_small]]


@pytest.fixture(scope="session")
def mol_sizes():
    return dict(N3=3, N5=5)


@pytest.fixture(scope="session")
def property_shapes(mol_size):
    return dict(
        prop1=[1],
        der1=[mol_size, 3],
        contrib1=[mol_size, 1],
        prop2=[1],
        der2=[mol_size, 3],
    )


@pytest.fixture(scope="session")
def properties(property_shapes):
    pass


@pytest.fixture(scope="session")
def property_shapes(mol_sizes):
    pass


@pytest.fixture(scope="session")
def ats(db_size, mol_sizes):
    n_small = np.random.randint(1, db_size - 1)
    molecules = []
    data = []
    for i in range(db_size):
        mol_type = int(i <= n_small)
        data.append(
            {
                "prop": np.random.rand(1),
                "der": np.random.rand(mol_sizes[mol_type], 3),
                "contrib": np.random.rand(mol_sizes[mol_type], 1),
            }
        )
        molecules.append(
            Atoms(
                "N{}".format(mol_sizes[mol_type]),
                np.random.rand(mol_sizes[mol_type], 3),
            )
        )
    return molecules, data


@pytest.fixture(scope="session")
def build_db(tmp_db_path, ats):
    molecules, data = ats
    with connect(tmp_db_path) as conn:
        for mol, properties in zip(molecules, data):
            conn.write(mol, data=properties)

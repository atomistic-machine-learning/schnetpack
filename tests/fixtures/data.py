import os
import pytest
import numpy as np
import schnetpack as spk
from ase import Atoms
from ase.db import connect


__all__ = [
    "tmp_db_path",
    "db_size",
    "n_small_molecules",
    "small_size",
    "big_size",
    "property_shapes",
    "properties",
    "properties1",
    "properties2",
    "ats",
    "build_db",
    "dataset",
    "split",
    "splits",
    "train",
    "val",
    "test",
    "batch_size",
    "train_loader",
    "val_loader",
    "test_loader",
    "shuffle",
]


# define settings
@pytest.fixture(scope="session")
def tmp_db_path(tmpdir_factory):
    return os.path.join(tmpdir_factory.mktemp("data"), "test2.db")


@pytest.fixture(scope="session")
def db_size():
    return 20


@pytest.fixture(scope="session")
def n_small_molecules(db_size):
    n_small = np.random.randint(1, db_size - 1)
    return n_small


@pytest.fixture(scope="session")
def small_size():
    return 3


@pytest.fixture(scope="session")
def big_size():
    return 5


@pytest.fixture(scope="session")
def property_shapes(small_size, big_size):
    return {
        "N{}".format(small_size): dict(
            prop1=[1],
            der1=[small_size, 3],
            contrib1=[small_size, 1],
            prop2=[1],
            der2=[small_size, 3],
        ),
        "N{}".format(big_size): dict(
            prop1=[1],
            der1=[big_size, 3],
            contrib1=[big_size, 1],
            prop2=[1],
            der2=[big_size, 3],
        ),
    }


@pytest.fixture(scope="session")
def properties(property_shapes):
    return list(list(property_shapes.values())[0].keys())


@pytest.fixture(scope="session")
def properties1(properties):
    return [prop for prop in properties if prop.endswith("1")]


@pytest.fixture(scope="session")
def properties2(properties):
    return [prop for prop in properties if prop.endswith("2")]


# generate random data
@pytest.fixture(scope="session")
def ats(db_size, n_small_molecules, small_size, big_size, property_shapes):
    mol_size = small_size
    molecules = []
    data = []
    for i in range(db_size):
        if i <= n_small_molecules:
            mol_size = big_size
        shapes = property_shapes["N{}".format(mol_size)]

        data.append({key: np.random.rand(*shape) for key, shape in shapes.items()})
        molecules.append(Atoms("N{}".format(mol_size), np.random.rand(mol_size, 3)))

    return molecules, data


# write data to db
@pytest.fixture(scope="session")
def build_db(tmp_db_path, ats):
    molecules, data = ats
    with connect(tmp_db_path) as conn:
        for mol, properties in zip(molecules, data):
            conn.write(mol, data=properties)


# dataset
@pytest.fixture(scope="session")
def dataset(build_db, tmp_db_path):
    return spk.data.AtomsData(dbpath=tmp_db_path)


@pytest.fixture(scope="session")
def split():
    return 10, 7


@pytest.fixture(scope="session")
def splits(dataset, split):
    return spk.data.train_test_split(dataset, *split)


@pytest.fixture(scope="session")
def train(splits):
    return splits[0]


@pytest.fixture(scope="session")
def val(splits):
    return splits[1]


@pytest.fixture(scope="session")
def test(splits):
    return splits[2]


# dataloader
@pytest.fixture(scope="session")
def batch_size():
    return 4


@pytest.fixture(scope="session")
def train_loader(train, batch_size):
    return spk.data.AtomsLoader(train, batch_size)


@pytest.fixture(scope="session")
def val_loader(val, batch_size):
    return spk.data.AtomsLoader(val, batch_size)


@pytest.fixture(scope="session")
def test_loader(test, batch_size):
    return spk.data.AtomsLoader(test, batch_size)


@pytest.fixture(scope="session")
def shuffle():
    return True

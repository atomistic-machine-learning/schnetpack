import os

import pytest
import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from ase.db import connect

import schnetpack as spk
from schnetpack import Properties

__all__ = ["n_atoms"]


@pytest.fixture
def n_atoms():
    return 19


@pytest.fixture
def n_spatial_basis():
    return 25


@pytest.fixture
def single_spatial_basis():
    return 1


@pytest.fixture
def n_filters():
    return 128


@pytest.fixture
def n_interactions():
    return 1


@pytest.fixture
def atomic_env(batchsize, n_atoms, n_filters):
    return torch.rand((batchsize, n_atoms, n_filters))


@pytest.fixture
def atomic_numbers(batchsize, n_atoms):
    atoms = np.random.randint(1, 9, (1, n_atoms))
    return torch.LongTensor(np.repeat(atoms, batchsize, axis=0))


@pytest.fixture
def atom_mask(atomic_numbers):
    return torch.ones(atomic_numbers.shape)


@pytest.fixture
def atomtypes(atomic_numbers):
    return set(atomic_numbers[0].data.numpy())


@pytest.fixture
def positions(batchsize, n_atoms):
    return torch.rand((batchsize, n_atoms, 3))


@pytest.fixture
def cell(batchsize):
    return torch.zeros((batchsize, 3, 3))


@pytest.fixture
def cell_offset(batchsize, n_atoms):
    return torch.zeros((batchsize, n_atoms, n_atoms - 1, 3))


@pytest.fixture
def neighbors(batchsize, n_atoms):
    neighbors = np.array([range(n_atoms)] * n_atoms)
    neighbors = neighbors[~np.eye(neighbors.shape[0], dtype=bool)].reshape(
        neighbors.shape[0], -1
    )[np.newaxis, :]
    return torch.LongTensor(np.repeat(neighbors, batchsize, axis=0))


@pytest.fixture
def neighbor_mask(batchsize, n_atoms):
    return torch.ones((batchsize, n_atoms, n_atoms - 1))


@pytest.fixture
def schnet_batch(
    atomic_numbers, atom_mask, positions, cell, cell_offset, neighbors, neighbor_mask
):
    inputs = {}
    inputs[Properties.Z] = atomic_numbers
    inputs[Properties.R] = positions
    inputs[Properties.cell] = cell
    inputs[Properties.cell_offset] = cell_offset
    inputs[Properties.neighbors] = neighbors
    inputs[Properties.neighbor_mask] = neighbor_mask
    inputs[Properties.atom_mask] = atom_mask
    return inputs


@pytest.fixture
def distances(batchsize, n_atoms):
    return torch.rand((batchsize, n_atoms, n_atoms - 1))


@pytest.fixture
def expanded_distances(batchsize, n_atoms, n_spatial_basis):
    return torch.rand((batchsize, n_atoms, n_atoms - 1, n_spatial_basis))


@pytest.fixture
def filter_network(single_spatial_basis, n_filters):
    return nn.Linear(single_spatial_basis, n_filters)


@pytest.fixture(scope="session")
def tmp_data_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


@pytest.fixture(scope="session")
def tmp_db_path(tmp_data_dir):
    return os.path.join(tmp_data_dir, "test2.db")


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


@pytest.fixture(scope="session")
def build_db(tmp_db_path, ats):
    molecules, data = ats
    with connect(tmp_db_path) as conn:
        for mol, properties in zip(molecules, data):
            conn.write(mol, data=properties)


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

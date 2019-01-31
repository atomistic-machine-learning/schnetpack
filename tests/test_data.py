import os

import numpy as np
import pytest
from ase import Atoms

import schnetpack.data


@pytest.fixture
def max_atoms():
    return 3


@pytest.fixture
def num_data():
    return 5


@pytest.fixture
def property_spec():
    spec = {
        'energy': (1,),
        'dipole_moment': (3,),
        'forces': (-1, 3)
    }
    return spec


@pytest.fixture
def empty_asedata(tmpdir, max_atoms, property_spec):
    return schnetpack.data.AtomsData(os.path.join(str(tmpdir), 'test.db'),
                                     required_properties=list(
                                         property_spec.keys()))


@pytest.fixture
def example_data(max_atoms, num_data):
    data = []
    for i in range(1, num_data + 1):
        n_atoms = min(max_atoms, i)
        z = np.random.randint(1, 100, size=(n_atoms,))
        r = np.random.randn(n_atoms, 3)
        c = np.random.randn(3, 3)
        pbc = np.random.randint(0, 2, size=(3,)) > 0
        ats = Atoms(numbers=z, positions=r, cell=c, pbc=pbc)

        props = {
            'energy': np.random.rand(1),
            'dipole_moment': np.random.rand(3),
            'forces': np.random.rand(n_atoms, 3)
        }
        data.append((ats, props))

    return data


@pytest.fixture
def example_asedata(tmpdir, max_atoms, property_spec, example_data):
    data = schnetpack.data.AtomsData(os.path.join(str(tmpdir), 'test.db'),
                                     required_properties=list(
                                         property_spec.keys()))
    # add data
    for ats, props in example_data:
        data.add_system(ats, **props)
    return data


def test_add_and_read(empty_asedata, example_data):
    # add data
    for ats, props in example_data:
        empty_asedata.add_system(ats, **props)

    assert len(empty_asedata) == len(example_data)
    assert os.path.exists(empty_asedata.dbpath)

    for i in range(len(example_data)):
        d = empty_asedata[i]
    return empty_asedata


def test_empty_subset_of_subset(empty_asedata, example_data):
    data = test_add_and_read(empty_asedata, example_data)
    subset = data.create_subset([0, 1])
    subset.create_subset([])


def test_merging(tmpdir, example_asedata):
    merged_dbpath = os.path.join(str(tmpdir), 'merged.db')

    schnetpack.data.merge_datasets(merged_dbpath,
                                   [example_asedata.dbpath,
                                    example_asedata.dbpath])

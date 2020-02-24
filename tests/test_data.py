import os
import torch
import numpy as np
import pytest
from ase import Atoms

import schnetpack as spk
import schnetpack.data

__all__ = ["max_atoms", "example_dataset", "property_spec", "example_data", "num_data"]


@pytest.fixture
def max_atoms():
    return 3


@pytest.fixture
def num_data():
    return 5


@pytest.fixture
def property_spec():
    spec = {"energy": (1,), "dipole_moment": (3,), "forces": (-1, 3)}
    return spec


@pytest.fixture
def empty_dataset(tmpdir, max_atoms, property_spec):
    return schnetpack.data.AtomsData(
        os.path.join(str(tmpdir), "test.db"),
        available_properties=list(property_spec.keys()),
    )


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
            "energy": np.array([5.0], dtype=np.float32),
            "dipole_moment": np.random.rand(3),
            "forces": np.random.rand(n_atoms, 3),
        }
        data.append((ats, props))

    return data


@pytest.fixture
def example_dataset(tmpdir, max_atoms, property_spec, example_data):
    data = schnetpack.data.AtomsData(
        os.path.join(str(tmpdir), "test.db"),
        available_properties=list(property_spec.keys()),
    )
    # add data
    for ats, props in example_data:
        data.add_system(ats, **props)
    return data


def test_add_and_read(empty_dataset, example_data):
    # add data
    for ats, props in example_data:
        empty_dataset.add_system(ats, **props)

    assert len(empty_dataset) == len(example_data)
    assert os.path.exists(empty_dataset.dbpath)

    for i in range(len(example_data)):
        d = empty_dataset[i]
    return empty_dataset


def test_empty_subset_of_subset(empty_dataset, example_data):
    data = test_add_and_read(empty_dataset, example_data)
    subset = data.create_subset([0, 1])
    subsubset = subset.create_subset([])
    assert len(subset) == 2
    assert len(subsubset) == 0


@pytest.fixture(params=[None, ["example1", "example2", "ex3"]])
def partition_names(request):
    return request.param


def test_merging(tmpdir, example_dataset, partition_names):
    # create merged dataset by repeating original three times
    merged_dbpath = os.path.join(str(tmpdir), "merged.db")

    parts = [example_dataset.dbpath, example_dataset.dbpath, example_dataset.dbpath]
    if partition_names is not None:
        parts = {k: v for k, v in zip(partition_names, parts)}

    merged_data = schnetpack.data.merge_datasets(merged_dbpath, parts)

    # check merged
    assert len(merged_data) == 3 * len(example_dataset)

    partitions = merged_data.get_metadata("partitions")
    partition_meta = merged_data.get_metadata("partition_meta")

    assert len(partitions) == 3

    for p in partitions.values():
        assert len(p) == 2

    if partition_names is not None:
        assert "example1" in partitions.keys()
        assert "example2" in partitions.keys()
        assert "ex3" in partitions.keys()


@pytest.fixture(params=[1, 10])
def batch_size(request):
    return request.param


def test_loader(example_dataset, batch_size):
    loader = schnetpack.data.AtomsLoader(example_dataset, batch_size)
    for batch in loader:
        for entry in batch.values():
            assert entry.shape[0] == min(batch_size, len(loader.dataset))

    mu, std = loader.get_statistics("energy")
    assert mu["energy"] == torch.FloatTensor([5.0])
    assert std["energy"] == torch.FloatTensor([0.0])


from tests.fixtures.qm9 import qm9_dbpath, qm9_avlailable_properties


def test_dataset(qm9_dbpath, qm9_avlailable_properties):
    # path exists and valid properties
    dataset = spk.data.AtomsData(
        qm9_dbpath, available_properties=qm9_avlailable_properties
    )
    assert dataset.available_properties == qm9_avlailable_properties
    assert dataset.__len__() == 19

    # test valid path exists but wrong properties
    too_many = qm9_avlailable_properties + ["invalid"]
    not_all = qm9_avlailable_properties[:-1]
    wrong_prop = qm9_avlailable_properties[:-1] + ["invalid"]
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.AtomsData(qm9_dbpath, available_properties=too_many)
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.AtomsData(qm9_dbpath, available_properties=not_all)
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.AtomsData(qm9_dbpath, available_properties=wrong_prop)

    # test valid path, but no properties
    dataset = spk.data.AtomsData(qm9_dbpath)
    assert set(dataset.available_properties) == set(qm9_avlailable_properties)


def test_extension_check():
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.atoms.AtomsData("test/path")


@pytest.fixture(scope="session")
def h2o():
    return Atoms(positions=np.random.rand(3, 3), numbers=[1, 1, 8])


@pytest.fixture(scope="session")
def o2():
    return Atoms(positions=np.random.rand(2, 3), numbers=[8, 8])


def test_get_center(h2o, o2):
    # test if centers are equal for symmetric molecule
    com = spk.data.get_center_of_mass(o2)
    cog = spk.data.get_center_of_geometry(o2)

    assert list(com.shape) == [3]
    assert list(cog.shape) == [3]
    np.testing.assert_array_almost_equal(com, cog)

    # test if centers are different for asymmetric molecule
    com = spk.data.get_center_of_mass(h2o)
    cog = spk.data.get_center_of_geometry(h2o)

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, com, cog)


def test_concatenation(example_dataset):
    len_e = len(example_dataset)
    # create subset
    subset = spk.data.create_subset(example_dataset, [0, 1])

    # create concat dataset
    concat = example_dataset + subset
    concat2 = concat + subset

    # test lengths
    assert len(concat) == len(subset) + len(example_dataset)
    assert len(subset) == 2
    assert len(concat2) == len(concat) + len(subset)

    for i in range(len_e):
        c, e = concat[i], example_dataset[i]
        for key in c.keys():
            if key == "_idx":
                continue
            cv, ev = c[key], e[key]
            assert torch.equal(cv, ev)

    for i in range(2):
        c, e = concat[len_e + i], subset[i]
        for key in c.keys():
            if key == "_idx":
                continue
            cv, ev = c[key], e[key]
            assert torch.equal(cv, ev), "key {} does not match!".format(key)

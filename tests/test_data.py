import os
import torch
import numpy as np
import pytest

import schnetpack as spk
from tests.assertions import assert_dataset_equal
from numpy.testing import assert_array_almost_equal

from tests.fixtures import *


def test_add_and_read(empty_dataset, example_data):
    """
    Test if data can be added to existing dataset.
    """
    # add data
    for ats, props in example_data:
        empty_dataset.add_system(ats, props)

    assert len(empty_dataset) == len(example_data)
    assert os.path.exists(empty_dataset.dbpath)


def test_empty_subset_of_subset(example_subset):
    """
    Test if empty subsubset can be created.
    """
    subsubset = spk.data.create_subset(example_subset, [])
    assert len(example_subset) == 2
    assert len(subsubset) == 0


def test_merging(tmpdir, example_dataset, partition_names):
    # create merged dataset by repeating original three times
    merged_dbpath = os.path.join(str(tmpdir), "merged.db")

    parts = [example_dataset.dbpath, example_dataset.dbpath, example_dataset.dbpath]
    if partition_names is not None:
        parts = {k: v for k, v in zip(partition_names, parts)}

    merged_data = spk.data.merge_datasets(merged_dbpath, parts)

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


def test_loader(example_loader, batch_size):
    """
    Test dataloader iteration and batch shapes.
    """
    for batch in example_loader:
        for entry in batch.values():
            assert entry.shape[0] == min(batch_size, len(example_loader.dataset))


def test_statistics_calculation(example_loader, dataset_stats, main_properties):
    """
    Test statistics calculation of dataloader.
    """
    means, stds = example_loader.get_statistics(main_properties)
    for pname in main_properties:
        assert_array_almost_equal(means[pname].numpy(), dataset_stats[0][pname])
        assert_array_almost_equal(stds[pname].numpy(), dataset_stats[1][pname])


def test_dataset(example_dataset, tmp_dbpath, available_properties, num_data):
    assert example_dataset.available_properties == available_properties
    assert example_dataset.__len__() == num_data

    # test valid path exists but wrong properties
    too_many = available_properties + ["invalid"]
    not_all = available_properties[:-1]
    wrong_prop = available_properties[:-1] + ["invalid"]
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.AtomsData(tmp_dbpath, available_properties=too_many)
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.AtomsData(tmp_dbpath, available_properties=not_all)
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.AtomsData(tmp_dbpath, available_properties=wrong_prop)

    # test valid path, but no properties
    dataset = spk.data.AtomsData(tmp_dbpath)
    assert set(dataset.available_properties) == set(available_properties)


def test_extension_check():
    """
    Test if dataset raises error if .db is missing  in dbpath.
    """
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.atoms.AtomsData("test/path")


def test_get_center(h2o, o2):
    """
    Test calculation of molecular centers.
    """
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


def test_concatenation(
    example_dataset, example_subset, example_concat_dataset, example_concat_dataset2
):
    """
    Test ids of concatenated datasets.
    """
    len_e = len(example_dataset)
    # test lengths
    assert len(example_concat_dataset) == len(example_subset) + len(example_dataset)
    assert len(example_subset) == 2
    assert len(example_concat_dataset2) == len(example_concat_dataset) + len(
        example_subset
    )

    for i in range(len_e):
        c, e = example_concat_dataset[i], example_dataset[i]
        for key in c.keys():
            if key == "_idx":
                continue
            cv, ev = c[key], e[key]
            assert torch.equal(cv, ev)

    for i in range(2):
        c, e = example_concat_dataset[len_e + i], example_subset[i]
        for key in c.keys():
            if key == "_idx":
                continue
            cv, ev = c[key], e[key]
            assert torch.equal(cv, ev), "key {} does not match!".format(key)


def test_save_concatenated(tmp_data_dir, example_concat_dataset):
    """
    Test if a concatenated dataset can be saved.

    """
    # save dataset two times
    tmp_dbpath1 = os.path.join(tmp_data_dir, "concat.db")
    spk.data.save_dataset(tmp_dbpath1, example_concat_dataset)
    tmp_dbpath2 = os.path.join(tmp_data_dir, "concat2.db")
    spk.data.save_dataset(tmp_dbpath2, example_concat_dataset)

    # check if paths exist
    assert os.path.exists(tmp_dbpath1)
    assert os.path.exists(tmp_dbpath2)

    # assert if saved datasets are equal
    dataset1 = spk.data.AtomsData(tmp_dbpath1)
    dataset2 = spk.data.AtomsData(tmp_dbpath2)

    assert_dataset_equal(dataset1, dataset2)


def test_qm9(qm9_path, qm9_dataset):
    """
    Test if QM9 dataset object has same behaviour as AtomsData.

    """
    atoms_data = spk.AtomsData(qm9_path)
    assert_dataset_equal(atoms_data, qm9_dataset)


def test_md17(ethanol_path, md17_dataset):
    """
    Test if MD17 dataset object has same behaviour as AtomsData.
    """
    atoms_data = spk.AtomsData(ethanol_path)
    assert_dataset_equal(atoms_data, md17_dataset)


def test_ani1(ani1_path, ani1_dataset):
    """
    Test if MD17 dataset object has same behaviour as AtomsData.
    """
    atoms_data = spk.AtomsData(ani1_path)
    assert_dataset_equal(atoms_data, ani1_dataset)

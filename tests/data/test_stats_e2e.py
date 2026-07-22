"""
End-to-end regression tests for training statistics (review finding S1).

The data-pipeline offset transform (RemoveOffsets) and the model
postprocessor (AddOffsets) must hold the identical raw-training-data
statistics, even when the postprocessor is initialized late — after the
train dataset already has its transforms attached. Historically the late
path recomputed statistics on already-transformed data, shifting every
prediction by a constant.
"""

import os

import numpy as np
import pytest
import torch
from ase import Atoms

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData, AtomsDataModule, estimate_atomrefs
from schnetpack.model import AtomisticModel
from schnetpack.transform import AddOffsets, RemoveOffsets

ENERGY = "energy"


@pytest.fixture
def stats_dbpath(tmp_path):
    """Small deterministic H/O dataset with known energies."""
    datapath = os.path.join(tmp_path, "stats_test.db")
    db = ASEAtomsData.create(
        datapath, distance_unit="Ang", property_unit_dict={ENERGY: "eV"}
    )

    rng = np.random.RandomState(42)
    atoms_list = []
    property_list = []
    for i in range(20):
        n_h = 1 + i % 4
        n_o = 1 + (i * 3) % 5
        numbers = [1] * n_h + [8] * n_o
        atoms_list.append(
            Atoms(numbers=numbers, positions=rng.randn(len(numbers), 3))
        )
        energy = -2.0 * n_h + 5.0 * n_o + 0.1 * (i % 7)
        property_list.append({ENERGY: np.array([energy])})

    db.add_systems(property_list, atoms_list)
    return datapath


def _setup_datamodule(datapath, tmp_path, transforms):
    dataset = ASEAtomsData(datapath, transforms=transforms)
    dm = AtomsDataModule(
        dataset,
        batch_size=5,
        num_train=10,
        num_val=5,
        num_test=5,
        split_file=os.path.join(tmp_path, "split.npz"),
        num_workers=0,
    )
    dm.setup()
    return dm


def _raw_train_subset(datapath, train_idx):
    """Independent raw view: fresh dataset instance, no transforms, no split."""
    return ASEAtomsData(datapath).subset(list(train_idx))


def _raw_per_atom_mean(datapath, train_idx):
    subset = _raw_train_subset(datapath, train_idx)
    per_atom = [
        subset[i][ENERGY] / subset[i][structure.n_atoms] for i in range(len(subset))
    ]
    return torch.cat(per_atom).mean()


def _late_initialized_postprocessor(dm, **offset_kwargs):
    """Initialize AddOffsets through the model path, after dm.setup()."""
    model = AtomisticModel(
        postprocessors=[AddOffsets(ENERGY, is_extensive=True, **offset_kwargs)]
    )
    model.initialize_transforms(dm)
    return model.postprocessors[0]


def test_offset_mean_consistent_with_late_postprocessor(
    stats_dbpath, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    remove_offsets = RemoveOffsets(ENERGY, remove_mean=True, is_extensive=True)
    dm = _setup_datamodule(stats_dbpath, tmp_path, [remove_offsets])

    # postprocessor initialized late: train transforms are already attached
    add_offsets = _late_initialized_postprocessor(dm, add_mean=True)

    expected_mean = _raw_per_atom_mean(stats_dbpath, dm.train_idx)

    assert torch.allclose(remove_offsets.mean.double(), expected_mean.double())
    assert torch.allclose(add_offsets.mean.double(), expected_mean.double())
    assert torch.allclose(add_offsets.mean, remove_offsets.mean)


def test_pipeline_actually_removes_raw_mean(stats_dbpath, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    remove_offsets = RemoveOffsets(ENERGY, remove_mean=True, is_extensive=True)
    dm = _setup_datamodule(stats_dbpath, tmp_path, [remove_offsets])

    raw_subset = _raw_train_subset(stats_dbpath, dm.train_idx)
    expected_mean = _raw_per_atom_mean(stats_dbpath, dm.train_idx)

    transformed = dm.train_dataset[0]
    raw = raw_subset[0]
    assert torch.allclose(
        transformed[ENERGY].double(),
        (raw[ENERGY] - expected_mean * raw[structure.n_atoms]).double(),
    )


def test_estimated_atomrefs_consistent_with_late_postprocessor(
    stats_dbpath, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    remove_offsets = RemoveOffsets(
        ENERGY,
        remove_mean=True,
        remove_atomrefs=True,
        estimate_atomref=True,
        is_extensive=True,
    )
    dm = _setup_datamodule(stats_dbpath, tmp_path, [remove_offsets])

    add_offsets = _late_initialized_postprocessor(
        dm, add_mean=True, add_atomrefs=True, estimate_atomref=True
    )

    expected_atomrefs = estimate_atomrefs(
        _raw_train_subset(stats_dbpath, dm.train_idx), is_extensive={ENERGY: True}
    )[ENERGY]

    assert torch.count_nonzero(expected_atomrefs) > 0
    assert torch.allclose(remove_offsets.atomref, expected_atomrefs)
    assert torch.allclose(add_offsets.atomref, expected_atomrefs)
    assert torch.allclose(add_offsets.mean, remove_offsets.mean)

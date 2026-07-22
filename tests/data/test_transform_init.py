"""
Unit tests for the unified transform initialization: `initialize(stats)`
takes a single stats source — any object with `get_stats`/`get_atomrefs`,
i.e. the datamodule's provider or the datamodule itself.
"""

import pytest
import torch

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData, AtomsDataModule
from schnetpack.data.provider import StatsAtomrefProvider
from schnetpack.transform import AddOffsets, RemoveOffsets, ScaleProperty

from .conftest import ENERGY, H_ATOMREF, O_ATOMREF

TRAIN_IDX = list(range(10))


def test_remove_offsets_reads_dataset_atomrefs_from_stats_source(
    stats_dbpath_with_atomrefs,
):
    provider = StatsAtomrefProvider(
        ASEAtomsData(stats_dbpath_with_atomrefs), TRAIN_IDX
    )
    transform = RemoveOffsets(
        ENERGY, remove_atomrefs=True, estimate_atomref=False, is_extensive=True
    )

    transform.initialize(provider)

    assert transform.atomref[1] == pytest.approx(H_ATOMREF)
    assert transform.atomref[8] == pytest.approx(O_ATOMREF)


def test_add_offsets_reads_dataset_atomrefs_from_stats_source(
    stats_dbpath_with_atomrefs,
):
    provider = StatsAtomrefProvider(
        ASEAtomsData(stats_dbpath_with_atomrefs), TRAIN_IDX
    )
    transform = AddOffsets(
        ENERGY, add_atomrefs=True, estimate_atomref=False, is_extensive=True
    )

    transform.initialize(provider)

    assert transform.atomref[1] == pytest.approx(H_ATOMREF)
    assert transform.atomref[8] == pytest.approx(O_ATOMREF)


def test_strict_atomref_initialization_fails_without_dataset_atomrefs(
    stats_dbpath,
):
    provider = StatsAtomrefProvider(ASEAtomsData(stats_dbpath), TRAIN_IDX)
    transform = RemoveOffsets(
        ENERGY, remove_atomrefs=True, estimate_atomref=False, is_extensive=True
    )

    with pytest.raises(RuntimeError, match=ENERGY):
        transform.initialize(provider)


def test_scale_property_initializes_from_stats_source(stats_dbpath):
    dataset = ASEAtomsData(stats_dbpath)
    provider = StatsAtomrefProvider(dataset, TRAIN_IDX)
    transform = ScaleProperty(input_key=ENERGY)

    transform.initialize(provider)

    per_atom = [
        dataset[i][ENERGY] / dataset[i][structure.n_atoms] for i in TRAIN_IDX
    ]
    expected_std = torch.cat(per_atom).double().std(correction=0)
    assert torch.allclose(transform.scale.double(), expected_std)


def test_datamodule_satisfies_the_stats_source_protocol(
    stats_dbpath_with_atomrefs, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    dm = AtomsDataModule(
        ASEAtomsData(stats_dbpath_with_atomrefs),
        batch_size=5,
        num_train=10,
        num_val=5,
        num_test=5,
        split_file=str(tmp_path / "split.npz"),
        num_workers=0,
    )
    dm.setup()

    transform = AddOffsets(
        ENERGY,
        add_mean=True,
        add_atomrefs=True,
        estimate_atomref=False,
        is_extensive=True,
    )
    transform.initialize(dm)

    assert transform.atomref[1] == pytest.approx(H_ATOMREF)
    mean, _std = dm.get_stats(ENERGY, True, True)
    assert torch.allclose(transform.mean, mean)

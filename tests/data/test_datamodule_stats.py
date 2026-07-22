"""
Tests for the datamodule's statistics plumbing: the train-partition
fingerprint and the on-disk statistics cache derived from the split file.
"""

import os

import schnetpack.data.provider
from schnetpack.data import ASEAtomsData, AtomsDataModule, calculate_stats
from schnetpack.data.provider import StatsAtomrefProvider

from .conftest import ENERGY


def _make_dm(datapath, split_file, num_train=10, **kwargs):
    return AtomsDataModule(
        ASEAtomsData(datapath),
        batch_size=5,
        num_train=num_train,
        num_val=5,
        split_file=str(split_file),
        num_workers=0,
        **kwargs,
    )


def _count_stats_calls(monkeypatch):
    calls = []

    def counting_calculate_stats(*args, **kwargs):
        calls.append(1)
        return calculate_stats(*args, **kwargs)

    monkeypatch.setattr(
        schnetpack.data.provider, "calculate_stats", counting_calculate_stats
    )
    return calls


def test_train_fingerprint_is_deterministic_across_create_and_load(
    stats_dbpath, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    split_file = tmp_path / "split.npz"

    dm_create = _make_dm(stats_dbpath, split_file)
    dm_create.setup()

    dm_load = _make_dm(stats_dbpath, split_file)
    dm_load.setup()

    assert isinstance(dm_create.train_fingerprint, str)
    assert len(dm_create.train_fingerprint) > 0
    assert dm_load.train_fingerprint == dm_create.train_fingerprint


def test_train_fingerprint_changes_with_partition(
    stats_dbpath, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    dm_a = _make_dm(stats_dbpath, tmp_path / "split_a.npz", num_train=10)
    dm_a.setup()
    dm_b = _make_dm(stats_dbpath, tmp_path / "split_b.npz", num_train=12)
    dm_b.setup()

    assert dm_a.train_fingerprint != dm_b.train_fingerprint


def test_second_datamodule_reads_stats_from_disk(
    stats_dbpath, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    calls = _count_stats_calls(monkeypatch)
    split_file = tmp_path / "split.npz"

    dm_first = _make_dm(stats_dbpath, split_file)
    dm_first.setup()
    first = dm_first.get_stats(ENERGY, True, False)
    assert len(calls) == 1

    dm_second = _make_dm(stats_dbpath, split_file)
    dm_second.setup()
    second = dm_second.get_stats(ENERGY, True, False)

    assert len(calls) == 1  # read from disk, not recomputed
    assert second[0] == first[0]
    assert second[1] == first[1]


def test_stats_file_written_next_to_split_file(stats_dbpath, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dm = _make_dm(stats_dbpath, tmp_path / "split.npz")
    dm.setup()
    dm.get_stats(ENERGY, True, False)

    assert os.path.exists(tmp_path / "split_stats.npz")


def test_fingerprint_mismatch_triggers_recompute(
    stats_dbpath, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    calls = _count_stats_calls(monkeypatch)
    stats_file = str(tmp_path / "stats.npz")
    dataset = ASEAtomsData(stats_dbpath)

    provider_a = StatsAtomrefProvider(
        dataset, list(range(10)), stats_file=stats_file
    )
    provider_a.get_stats(ENERGY, True, False)
    assert len(calls) == 1

    # same partition: served from disk
    provider_same = StatsAtomrefProvider(
        dataset, list(range(10)), stats_file=stats_file
    )
    provider_same.get_stats(ENERGY, True, False)
    assert len(calls) == 1

    # different partition: stored entries are invalid, recompute
    provider_b = StatsAtomrefProvider(
        dataset, list(range(12)), stats_file=stats_file
    )
    provider_b.get_stats(ENERGY, True, False)
    assert len(calls) == 2


def test_stats_file_none_disables_persistence(stats_dbpath, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = _count_stats_calls(monkeypatch)
    split_file = tmp_path / "split.npz"

    dm_first = _make_dm(stats_dbpath, split_file, stats_file=None)
    dm_first.setup()
    dm_first.get_stats(ENERGY, True, False)

    assert not os.path.exists(tmp_path / "split_stats.npz")

    dm_second = _make_dm(stats_dbpath, split_file, stats_file=None)
    dm_second.setup()
    dm_second.get_stats(ENERGY, True, False)
    assert len(calls) == 2  # nothing persisted, so it recomputes

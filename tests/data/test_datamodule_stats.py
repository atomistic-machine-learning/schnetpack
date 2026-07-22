"""
Tests for the datamodule's statistics plumbing: the train-partition
fingerprint and the on-disk statistics cache derived from the split file.
"""

from schnetpack.data import ASEAtomsData, AtomsDataModule


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

"""
Unit tests for StatsAtomrefProvider's pure interface: constructed from the
base dataset plus an explicit train index list, it must serve raw-data
statistics no matter what transforms are attached to the dataset.
"""

import numpy as np
import pytest
import torch

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData
from schnetpack.data.provider import StatsAtomrefProvider
from schnetpack.transform import RemoveOffsets

from .conftest import ENERGY, H_ATOMREF, O_ATOMREF

TRAIN_IDX = list(range(10))


def _provider_on_transformed_base(datapath):
    base = ASEAtomsData(
        datapath,
        transforms=[
            RemoveOffsets(
                ENERGY,
                remove_mean=True,
                is_extensive=True,
                property_mean=torch.tensor([100.0]),
            )
        ],
    )
    return StatsAtomrefProvider(base, TRAIN_IDX)


def test_provider_serves_raw_stats_for_train_indices(stats_dbpath):
    provider = _provider_on_transformed_base(stats_dbpath)

    mean, _std = provider.get_stats(ENERGY, True, False)

    raw = ASEAtomsData(stats_dbpath)
    per_atom = [raw[i][ENERGY] / raw[i][structure.n_atoms] for i in TRAIN_IDX]
    expected = torch.cat(per_atom).mean()
    assert torch.allclose(mean.double(), expected.double())


def test_provider_estimates_atomrefs_on_raw_train_indices(stats_dbpath):
    provider = _provider_on_transformed_base(stats_dbpath)

    atomrefs = provider.get_atomrefs(ENERGY, True)[ENERGY]

    from schnetpack.data import estimate_atomrefs

    expected = estimate_atomrefs(
        ASEAtomsData(stats_dbpath), is_extensive={ENERGY: True}, indices=TRAIN_IDX
    )[ENERGY]
    assert torch.count_nonzero(expected) > 0
    assert torch.allclose(atomrefs, expected)


def test_get_atomrefs_strict_returns_dataset_values(stats_dbpath_with_atomrefs):
    provider = StatsAtomrefProvider(ASEAtomsData(stats_dbpath_with_atomrefs), TRAIN_IDX)

    refs = provider.get_atomrefs(ENERGY, True, estimate=False)[ENERGY]

    assert refs[1] == pytest.approx(H_ATOMREF)
    assert refs[8] == pytest.approx(O_ATOMREF)


def test_get_atomrefs_strict_raises_without_dataset_values(stats_dbpath):
    provider = StatsAtomrefProvider(ASEAtomsData(stats_dbpath), TRAIN_IDX)

    with pytest.raises(RuntimeError, match=ENERGY):
        provider.get_atomrefs(ENERGY, True, estimate=False)


def test_datamodule_get_atomrefs_passes_estimate_through(
    stats_dbpath, tmp_path, monkeypatch
):
    from schnetpack.lightning import AtomsDataModule

    monkeypatch.chdir(tmp_path)
    dm = AtomsDataModule(
        ASEAtomsData(stats_dbpath),
        batch_size=5,
        num_train=10,
        num_val=5,
        num_test=5,
        split_file=str(tmp_path / "split.npz"),
        num_workers=0,
    )
    dm.setup()

    with pytest.raises(RuntimeError, match=ENERGY):
        dm.get_atomrefs(ENERGY, True, estimate=False)


def test_other_stats_sources_store_entries_through_the_provider(
    stats_dbpath, tmp_path, monkeypatch
):
    """An entry is computed once per train partition and then read back, also
    by another provider on the same stats file (a rerun, another rank)."""
    monkeypatch.chdir(tmp_path)  # the provider's lock lives in the working dir
    stats_file = str(tmp_path / "split_stats.npz")
    calls = []

    def compute():
        calls.append(1)
        return np.array([1.5, -2.0])

    providers = [
        StatsAtomrefProvider(ASEAtomsData(stats_dbpath), TRAIN_IDX, stats_file)
        for _ in range(2)
    ]
    values = [p.read_or_compute("teacher:abc:stats", compute) for p in providers]

    assert len(calls) == 1
    for value in values:
        np.testing.assert_array_equal(value, [1.5, -2.0])

"""
Unit tests for StatsAtomrefProvider's pure interface: constructed from the
base dataset plus an explicit train index list, it must serve raw-data
statistics no matter what transforms are attached to the dataset.
"""

import torch

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData
from schnetpack.data.provider import StatsAtomrefProvider
from schnetpack.transform import RemoveOffsets

from .conftest import ENERGY

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

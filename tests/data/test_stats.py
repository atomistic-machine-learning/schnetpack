"""
Unit tests for the explicit-indices interface of the statistics functions.

`calculate_stats` and `estimate_atomrefs` accept the dataset plus an explicit
index list and must compute on the raw data — regardless of any transforms or
split label attached to the dataset they are handed.
"""

import torch

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData, calculate_stats, estimate_atomrefs
from schnetpack.transform import RemoveOffsets

from .conftest import ENERGY


def _shift_transform(offset):
    """A self-contained transform that shifts the energy on every access."""
    return RemoveOffsets(
        ENERGY,
        remove_mean=True,
        is_extensive=True,
        property_mean=torch.tensor([offset]),
    )


def _transformed_view(datapath):
    """A dataset view with both plain and split-specific transforms attached."""
    base = ASEAtomsData(
        datapath,
        transforms=[_shift_transform(100.0)],
        train_transforms=[_shift_transform(200.0)],
    )
    return base.subset(list(range(len(base))), split="train")


def _manual_per_atom_mean(datapath, indices):
    raw = ASEAtomsData(datapath)
    per_atom = [raw[i][ENERGY] / raw[i][structure.n_atoms] for i in indices]
    return torch.cat(per_atom).mean()


def test_calculate_stats_with_indices_yields_raw_statistics(stats_dbpath):
    view = _transformed_view(stats_dbpath)
    indices = list(range(10))

    mean, _std = calculate_stats(view, divide_by_atoms={ENERGY: True}, indices=indices)[
        ENERGY
    ]

    expected = _manual_per_atom_mean(stats_dbpath, indices)
    expected_all = _manual_per_atom_mean(stats_dbpath, range(len(view)))
    assert not torch.allclose(expected, expected_all)  # indices must matter
    assert torch.allclose(mean.double(), expected.double())


def test_estimate_atomrefs_with_indices_yields_raw_estimates(stats_dbpath):
    view = _transformed_view(stats_dbpath)
    indices = list(range(10))

    atomrefs = estimate_atomrefs(view, is_extensive={ENERGY: True}, indices=indices)[
        ENERGY
    ]

    raw_subset = ASEAtomsData(stats_dbpath).subset(indices)
    expected = estimate_atomrefs(raw_subset, is_extensive={ENERGY: True})[ENERGY]

    assert torch.count_nonzero(expected) > 0
    assert torch.allclose(atomrefs, expected)

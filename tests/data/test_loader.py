import torch

import schnetpack.properties as structure
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import ASENeighborList


def test_collate_noenv(single_atom, two_atoms):
    batch = [single_atom, two_atoms]
    collated_batch = _atoms_collate_fn(batch)
    assert all([key in collated_batch.keys() for key in single_atom])
    assert structure.idx_m in collated_batch.keys()
    assert (collated_batch[structure.idx_m] == torch.tensor((0, 1, 1))).all()


def test_collate_env(single_atom, two_atoms):
    nll = ASENeighborList(cutoff=5.0)
    batch = [nll(single_atom), nll(two_atoms)]

    collated_batch = _atoms_collate_fn(batch)
    assert all([key in collated_batch.keys() for key in single_atom])
    assert structure.idx_m in collated_batch.keys()
    assert (collated_batch[structure.idx_m] == torch.tensor((0, 1, 1))).all()
    assert (collated_batch[structure.idx_i] == torch.tensor((1, 2))).all(), (
        collated_batch[structure.idx_i]
    )
    assert (collated_batch[structure.idx_j] == torch.tensor((2, 1))).all(), (
        collated_batch[structure.idx_j]
    )

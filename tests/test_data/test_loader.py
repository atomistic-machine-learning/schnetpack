import torch

from schnetpack.data.transforms import ASENeighborList
from schnetpack.data.loader import _atoms_collate_fn, AtomsLoader
from schnetpack import Structure


def test_collate_noenv(single_atom, two_atoms):
    batch = [single_atom, two_atoms]
    collated_batch = _atoms_collate_fn(batch)
    assert all([key in collated_batch.keys() for key in single_atom])
    assert Structure.seg_m in collated_batch.keys()
    assert (collated_batch[Structure.seg_m] == torch.tensor((0, 1, 3))).all()


def test_collate_env(single_atom, two_atoms):
    nll = ASENeighborList(cutoff=5.0)
    batch = [nll(single_atom), nll(two_atoms)]

    collated_batch = _atoms_collate_fn(batch)
    assert all([key in collated_batch.keys() for key in single_atom])
    assert Structure.seg_m in collated_batch.keys()
    assert (collated_batch[Structure.seg_m] == torch.tensor((0, 1, 3))).all()
    assert (
        collated_batch[Structure.idx_i] == torch.tensor((1, 2))
    ).all(), collated_batch[Structure.idx_i]
    assert (
        collated_batch[Structure.idx_j] == torch.tensor((2, 1))
    ).all(), collated_batch[Structure.idx_j]

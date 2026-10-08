"""The neighbor list a student shares with its teacher covers both cutoffs, each
given in its model's own length unit."""

import pytest
import torch
from ase.build import molecule

from schnetpack import properties
from schnetpack.transform import DistillationNeighborList, MatScipyNeighborList


def benzene():
    atoms = molecule("C6H6")
    return {
        properties.Z: torch.tensor(atoms.numbers),
        properties.R: torch.tensor(atoms.positions),
        properties.cell: torch.zeros(1, 3, 3, dtype=torch.float64),
        properties.pbc: torch.zeros(3, dtype=torch.bool),
    }


def n_pairs(neighbor_list):
    return neighbor_list(benzene())[properties.idx_i].shape[0]


def test_a_larger_teacher_cutoff_is_converted_and_widens_the_list():
    neighbor_list = DistillationNeighborList(
        MatScipyNeighborList(cutoff=2.0),
        teacher_cutoff=0.3,
        teacher_distance_unit="nm",
        student_distance_unit="Ang",
    )

    assert neighbor_list.cutoff == pytest.approx(3.0)
    assert n_pairs(neighbor_list) == n_pairs(MatScipyNeighborList(cutoff=3.0))
    assert n_pairs(neighbor_list) > n_pairs(MatScipyNeighborList(cutoff=2.0))


def test_a_smaller_teacher_cutoff_keeps_the_students():
    neighbor_list = DistillationNeighborList(
        MatScipyNeighborList(cutoff=3.0),
        teacher_cutoff=0.2,
        teacher_distance_unit="nm",
        student_distance_unit="Ang",
    )

    assert neighbor_list.cutoff == pytest.approx(3.0)
    assert n_pairs(neighbor_list) == n_pairs(MatScipyNeighborList(cutoff=3.0))

"""``prune_neighbors``: a neighbor list built at a larger cutoff, restricted to a
smaller one, must be the list a fresh build at the smaller cutoff gives."""

import numpy as np
import pytest
import torch
from ase import Atoms

import schnetpack.transform
from schnetpack import properties
from schnetpack.data import prune_neighbors
from schnetpack.interfaces.ase_interface import AtomsConverter
from schnetpack.transform import CollectAtomTriples, MatScipyNeighborList

CUTOFF = 3.0
WIDER = 4.0


def structures(pbc):
    rng = np.random.default_rng(0)
    return [
        Atoms(
            numbers=[6] * 10,
            positions=rng.uniform(0.0, 7.0, size=(10, 3)),
            cell=7.0 * np.eye(3),
            pbc=pbc,
        )
        for _ in range(3)
    ]


def built_at(cutoff, atoms, **kwargs):
    return AtomsConverter(
        neighbor_list=MatScipyNeighborList(cutoff=cutoff),
        dtype=torch.float64,
        **kwargs,
    )(atoms)


def pairs(inputs):
    """The pairs of a neighbor list, in an order-independent form."""
    table = np.column_stack(
        [
            inputs[properties.idx_i].numpy(),
            inputs[properties.idx_j].numpy(),
            inputs[properties.offsets].numpy().round(6),
        ]
    )
    return table[np.lexsort(table.T[::-1])]


def triples(inputs):
    """The atom triples of a neighbor list, read through its pairs. A triple's
    two neighbors are an unordered pair, whose order follows the list's."""
    idx_j = inputs[properties.idx_j]
    neighbors = np.sort(
        np.column_stack(
            [
                idx_j[inputs[properties.idx_j_triples]].numpy(),
                idx_j[inputs[properties.idx_k_triples]].numpy(),
            ]
        ),
        axis=1,
    )
    table = np.column_stack([inputs[properties.idx_i_triples].numpy(), neighbors])
    return table[np.lexsort(table.T[::-1])]


@pytest.mark.parametrize("pbc", [False, True], ids=["free", "periodic"])
def test_a_wider_list_pruned_to_the_cutoff_is_the_list_built_at_it(pbc):
    atoms = structures(pbc)
    wide = built_at(WIDER, atoms)

    pruned = prune_neighbors(wide, wide[properties.R], CUTOFF)

    np.testing.assert_array_equal(pairs(pruned), pairs(built_at(CUTOFF, atoms)))
    assert len(pruned[properties.idx_i]) < len(wide[properties.idx_i])


def test_triples_are_renumbered_onto_the_kept_pairs():
    atoms = structures(pbc=False)
    wide = built_at(WIDER, atoms, transforms=[CollectAtomTriples()])

    pruned = prune_neighbors(wide, wide[properties.R], CUTOFF)

    reference = built_at(CUTOFF, atoms, transforms=[CollectAtomTriples()])
    np.testing.assert_array_equal(triples(pruned), triples(reference))


def test_only_the_neighbor_entries_come_back_with_distances_on_request():
    wide = built_at(WIDER, structures(pbc=False))
    positions = wide[properties.R]

    pruned = prune_neighbors(wide, positions, CUTOFF)
    with_distances = prune_neighbors(wide, positions, CUTOFF, with_distances=True)

    n_kept = len(pruned[properties.idx_i])
    assert all(len(value) == n_kept for value in pruned.values())
    assert not {properties.R, properties.Z, properties.idx_m} & set(pruned)
    assert properties.Rij not in pruned
    idx_i, idx_j = with_distances[properties.idx_i], with_distances[properties.idx_j]
    torch.testing.assert_close(
        with_distances[properties.Rij],
        positions[idx_j] - positions[idx_i] + with_distances[properties.offsets],
    )


def test_the_transforms_keep_no_alias():
    """It works on collated batches, so it lives with the collate function
    (ADR-0028), not with the per-sample transforms."""
    assert not hasattr(schnetpack.transform, "prune_neighbors")

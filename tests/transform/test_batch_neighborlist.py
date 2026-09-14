"""Behaviour of ``BatchNeighborList``, the neighbor lists of a batch that keeps moving.

The reference throughout is a plain neighbor list built from scratch for the current
positions: whatever the module decides to reuse, what reaches the model has to be the
same set of pairs it would have got from a fresh build.
"""

from typing import Dict, List

import numpy as np
import pytest
import torch
from ase import Atoms

import schnetpack as spk
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn, split_batch
from schnetpack.interfaces.ase_interface import AtomsConverter, atoms_to_batch

CUTOFF = 3.0
CUTOFF_SKIN = 0.5
N_ATOMS = 12


def make_structures(
    pbc: bool, n_structures: int = 3, seed: int = 1, cell: bool = True
) -> List[Atoms]:
    """Structures of ``N_ATOMS`` carbons scattered through an 8 Angstrom box.

    ``cell=False`` leaves them without one, for the tests that move a structure far from
    where it started: the offsets of a pair list are cell shifts, so an atom outside its
    cell breaks ``R[j] - R[i] + offsets`` -- the convention ``PairwiseDistances`` and
    every neighbor list in schnetpack share.
    """
    rng = np.random.default_rng(seed)
    return [
        Atoms(
            numbers=[6] * N_ATOMS,
            positions=rng.uniform(0.0, 8.0, size=(N_ATOMS, 3)),
            cell=8.0 * np.eye(3) if cell else None,
            pbc=pbc,
        )
        for _ in range(n_structures)
    ]


def make_batch_neighbor_list(**kwargs) -> spk.transform.BatchNeighborList:
    return spk.transform.BatchNeighborList(
        neighbor_list=spk.transform.MatScipyNeighborList(cutoff=CUTOFF),
        cutoff_skin=CUTOFF_SKIN,
        dtype=torch.float64,
        **kwargs,
    )


def batch_of(structures: List[Atoms]) -> Dict[str, torch.Tensor]:
    """The structures as a batch, neighbor lists left to the module under test."""
    return atoms_to_batch(structures, dtype=torch.float64)


def freshly_built(structures: List[Atoms], **kwargs) -> Dict[str, torch.Tensor]:
    """The same structures through a plain neighbor list, built from scratch."""
    return AtomsConverter(
        neighbor_list=spk.transform.MatScipyNeighborList(cutoff=CUTOFF),
        dtype=torch.float64,
        **kwargs,
    )(structures)


def neighbor_pairs(inputs: Dict[str, torch.Tensor]) -> np.ndarray:
    """The neighbor list of a batch, in an order-independent, comparable form."""
    pairs = np.column_stack(
        [
            inputs[properties.idx_i].cpu().numpy(),
            inputs[properties.idx_j].cpu().numpy(),
            inputs[properties.offsets].cpu().numpy().round(6),
        ]
    )
    return pairs[np.lexsort(pairs.T[::-1])]


def displaced(structures: List[Atoms], scale: float, seed: int = 2) -> List[Atoms]:
    rng = np.random.default_rng(seed)
    moved = []
    for structure in structures:
        structure = structure.copy()
        structure.positions += rng.normal(scale=scale, size=(N_ATOMS, 3))
        moved.append(structure)
    return moved


@pytest.mark.parametrize("pbc", [False, True], ids=["free", "periodic"])
@pytest.mark.parametrize(
    "displacement, path",
    # half the skin is the threshold, so these pick the reuse and the rebuild branch
    [(0.1 * CUTOFF_SKIN, "reuse"), (2.0 * CUTOFF_SKIN, "rebuild")],
    ids=["reuse", "rebuild"],
)
def test_update_never_returns_pairs_beyond_the_cutoff(pbc, displacement, path):
    """Reusing a list must not leak the skin into what the model sees.

    The list is built out to ``cutoff + cutoff_skin`` so that it stays valid while the
    atoms move; the pairs beyond the cutoff have to be dropped again before the batch
    reaches the model, on the step that rebuilds the list and on every step that reuses
    it. Checked against a plain neighbor list built for the same positions.
    """
    structures = make_structures(pbc)
    neighbor_list = make_batch_neighbor_list()
    neighbor_list.update(batch_of(structures))

    moved = displaced(structures, displacement)
    updated = neighbor_list.update(batch_of(moved))

    np.testing.assert_allclose(
        neighbor_pairs(updated), neighbor_pairs(freshly_built(moved)), atol=1e-6
    )

    # and the branch under test really is the one that ran
    rebuilt = torch.allclose(
        neighbor_list._references[0][properties.R],
        torch.from_numpy(moved[0].positions),
    )
    assert rebuilt == (path == "rebuild")


def test_update_accepts_a_batch_without_a_sample_index():
    """A batch read back from a trajectory has no sample index; it must still update.

    ``BatchwiseTrajectoryReader.frame`` stores no ``idx``, so a relaxation resumed from
    a frame would otherwise fail on its very first step.
    """
    structures = make_structures(pbc=False)
    inputs = batch_of(structures)
    del inputs[properties.idx]

    updated = make_batch_neighbor_list().update(inputs)

    assert updated[properties.idx_i].shape == updated[properties.idx_j].shape
    assert len(updated[properties.idx_i]) > 0


def test_only_the_structures_that_moved_are_rebuilt():
    """Each structure carries its own list and its own reference positions.

    A batch is rebuilt structure by structure, so one structure walking away must not
    cost its neighbors in the batch a rebuild -- nor touch the positions their own
    lists were measured against.
    """
    structures = make_structures(pbc=False)
    neighbor_list = make_batch_neighbor_list()
    neighbor_list.update(batch_of(structures))
    references = {
        idx: neighbor_list._references[idx][properties.R].clone() for idx in range(3)
    }

    moved = [structure.copy() for structure in structures]
    moved[1].positions += 2.0 * CUTOFF_SKIN

    assert neighbor_list._stale_structures(batch_of(moved)) == [1]

    neighbor_list.update(batch_of(moved))
    for idx in (0, 2):
        assert torch.allclose(
            references[idx], neighbor_list._references[idx][properties.R]
        )
    assert not torch.allclose(references[1], neighbor_list._references[1][properties.R])


def test_drift_is_measured_against_the_positions_each_list_was_built_for():
    """A structure creeping along must not have its budget reset by the others.

    Its drift is measured against the positions its own list was built for, so it is
    rebuilt once the total displacement since then passes half the skin -- however many
    steps that took, and whatever the rest of the batch did in the meantime.
    """
    structures = make_structures(pbc=False, cell=False)
    neighbor_list = make_batch_neighbor_list()

    current = [structure.copy() for structure in structures]
    neighbor_list.update(batch_of(current))

    # a quarter of the threshold per step for the creeper, well past it for its neighbor,
    # so that the other structure forces a rebuild on nearly every step
    step = 0.125 * CUTOFF_SKIN
    for _ in range(8):
        current[0].positions[:, 0] += step
        current[2].positions += 2.0 * CUTOFF_SKIN

        updated = neighbor_list.update(batch_of(current))
        np.testing.assert_allclose(
            neighbor_pairs(updated), neighbor_pairs(freshly_built(current)), atol=1e-6
        )


def test_a_batch_of_a_different_size_is_not_reused():
    """Cached lists are keyed by position in the batch, so a new batch starts over."""
    neighbor_list = make_batch_neighbor_list()
    neighbor_list.update(batch_of(make_structures(pbc=False, n_structures=3)))

    structures = make_structures(pbc=False, n_structures=5, seed=7)
    updated = neighbor_list.update(batch_of(structures))

    np.testing.assert_allclose(
        neighbor_pairs(updated), neighbor_pairs(freshly_built(structures)), atol=1e-6
    )


def test_a_changed_cell_forces_a_rebuild():
    """Positions can sit still while the cell moves the periodic images."""
    structures = make_structures(pbc=True)
    neighbor_list = make_batch_neighbor_list()
    neighbor_list.update(batch_of(structures))

    squeezed = [structure.copy() for structure in structures]
    for structure in squeezed:
        structure.set_cell(6.5 * np.eye(3))

    updated = neighbor_list.update(batch_of(squeezed))

    np.testing.assert_allclose(
        neighbor_pairs(updated), neighbor_pairs(freshly_built(squeezed)), atol=1e-6
    )


def test_caller_entries_survive_a_rebuild():
    """``update`` refreshes the neighborhoods and leaves everything else alone.

    A batch carries more than its structures -- energies, convergence flags, whatever
    the caller put there -- and dropping those on the steps that happen to rebuild is
    the kind of asymmetry that only shows up much later.
    """
    structures = make_structures(pbc=False)
    neighbor_list = make_batch_neighbor_list()
    neighbor_list.update(batch_of(structures))

    inputs = batch_of(displaced(structures, 2.0 * CUTOFF_SKIN))
    inputs["converged"] = torch.tensor([True, False, True])
    inputs[properties.energy] = torch.zeros(3, dtype=torch.float64)

    updated = neighbor_list.update(inputs)

    assert updated["converged"].tolist() == [True, False, True]
    assert properties.energy in updated
    assert properties.Z in updated and properties.n_atoms in updated


def test_neighbors_returns_no_structure_entries():
    """The MD calculator merges the result into the batch it is propagating.

    Handing back positions -- copies, at that -- would overwrite the live ones.
    """
    structures = make_structures(pbc=False)

    neighbors = make_batch_neighbor_list().neighbors(batch_of(structures))

    for key in (properties.R, properties.Z, properties.cell, properties.pbc):
        assert key not in neighbors
    assert properties.idx_i in neighbors and properties.offsets in neighbors


def test_triples_are_renumbered_onto_the_pruned_pairs():
    """``idx_j_triples`` indexes pairs, so pruning pairs has to renumber the triples.

    Left alone they would point at the wrong pairs, or past the end of the array.
    """
    structures = make_structures(pbc=False)
    neighbor_list = make_batch_neighbor_list(
        transforms=spk.transform.CollectAtomTriples()
    )

    updated = neighbor_list.update(batch_of(structures))
    reference = freshly_built(structures, transforms=spk.transform.CollectAtomTriples())

    n_pairs = len(updated[properties.idx_i])
    for key in (properties.idx_j_triples, properties.idx_k_triples):
        assert int(updated[key].max()) < n_pairs

    # the triples are the same ones a fresh build finds, read through the pair arrays
    def triple_atoms(inputs):
        idx_j = inputs[properties.idx_j]
        triples = np.column_stack(
            [
                inputs[properties.idx_i_triples].cpu().numpy(),
                idx_j[inputs[properties.idx_j_triples]].cpu().numpy(),
                idx_j[inputs[properties.idx_k_triples]].cpu().numpy(),
            ]
        )
        return triples[np.lexsort(triples.T[::-1])]

    np.testing.assert_array_equal(triple_atoms(updated), triple_atoms(reference))


def test_split_batch_inverts_the_collate_function():
    """Cutting a batch apart and putting it back must give the batch back."""
    structures = make_structures(pbc=True, n_structures=4)
    inputs = batch_of(structures)

    recollated = _atoms_collate_fn(split_batch(inputs))

    for key in (properties.n_atoms, properties.Z, properties.R, properties.pbc):
        assert torch.equal(recollated[key], inputs[key])
    assert torch.allclose(recollated[properties.cell], inputs[properties.cell])


def test_split_batch_handles_ragged_and_single_atom_structures():
    """Atom-wise and structure-wise entries are told apart by more than their length."""
    structures = [
        Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4, pbc=True),
        Atoms(numbers=[8, 1], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        Atoms(numbers=[2], positions=[[2.0, 0.0, 0.0]]),
    ]
    samples = split_batch(atoms_to_batch(structures, dtype=torch.float64))

    assert [int(s[properties.n_atoms]) for s in samples] == [1, 2, 1]
    assert [s[properties.idx].item() for s in samples] == [0, 1, 2]
    for sample, structure in zip(samples, structures):
        assert sample[properties.cell].shape == (1, 3, 3)
        assert sample[properties.pbc].shape == (1, 3)
        np.testing.assert_allclose(
            sample[properties.R].numpy(), structure.positions, atol=1e-6
        )


def test_split_batch_of_single_atom_structures_round_trips():
    """The one case where atom count and structure count coincide."""
    structures = [
        Atoms(numbers=[1], positions=[[float(i), 0.0, 0.0]], cell=np.eye(3) * 4)
        for i in range(3)
    ]
    inputs = atoms_to_batch(structures, dtype=torch.float64)

    recollated = _atoms_collate_fn(split_batch(inputs))

    assert torch.equal(recollated[properties.Z], inputs[properties.Z])
    assert torch.allclose(recollated[properties.cell], inputs[properties.cell])

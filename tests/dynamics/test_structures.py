import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import DatasetStructures, StatisticsStructures
from schnetpack.generative import MASK, GaussianPrior, MaskPrior
from schnetpack.transform import AllToAllNeighborList


def molecule(z):
    """One structure as a dataset item: atom types, positions, count."""
    z = torch.tensor(z)
    return {
        properties.Z: z,
        properties.R: torch.arange(3.0 * len(z)).reshape(-1, 3),
        properties.n_atoms: torch.tensor([len(z)]),
    }


DATASET = [molecule([1, 6]), molecule([8, 1, 1]), molecule([6, 6, 1, 1])]


def test_dataset_structures_collate_like_the_dataloader():
    batch = DatasetStructures(DATASET, shuffle=False)(2)
    assert torch.equal(batch[properties.Z], torch.tensor([1, 6, 8, 1, 1]))
    assert torch.equal(batch[properties.n_atoms], torch.tensor([2, 3]))
    assert torch.equal(batch[properties.idx_m], torch.tensor([0, 0, 1, 1, 1]))
    assert batch[properties.R].shape == (5, 3)


def test_dataset_structures_draw_every_structure_once_per_pass():
    torch.manual_seed(0)
    structures = DatasetStructures(DATASET)
    first = structures(2)[properties.n_atoms].tolist()
    second = structures(1)[properties.n_atoms].tolist()
    assert sorted(first + second) == [2, 3, 4]
    # the next pass starts over
    assert structures(3)[properties.n_atoms].shape == (3,)


def test_all_to_all_on_the_batch_matches_the_collated_per_structure_lists():
    data = [*DATASET, molecule([1]), molecule([7, 1])]  # one without pairs
    nl = AllToAllNeighborList()
    per_structure = DatasetStructures([nl(dict(m)) for m in data], shuffle=False)
    on_batch = DatasetStructures(data, shuffle=False, transforms=[nl])
    expected, batch = per_structure(len(data)), on_batch(len(data))
    for key in (
        properties.idx_i,
        properties.idx_j,
        properties.lidx_i,
        properties.lidx_j,
        properties.offsets,
    ):
        assert torch.equal(batch[key], expected[key]), key


def test_dataset_structures_in_order_wrap_around():
    batch = DatasetStructures(DATASET, shuffle=False)(5)
    assert batch[properties.n_atoms].tolist() == [2, 3, 4, 2, 3]


def test_statistics_structures_follow_the_histograms():
    torch.manual_seed(0)
    structures = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 0, 1]),
        atom_types=torch.tensor([0, 0, 0, 0, 0, 0, 1]),
    )
    batch = structures(4)
    assert batch[properties.n_atoms].tolist() == [3, 3, 3, 3]
    assert set(batch[properties.Z].tolist()) == {6}
    assert torch.equal(batch[properties.idx_m], torch.arange(4).repeat_interleave(3))
    assert properties.R not in batch
    assert batch[properties.cell].shape == (4, 3, 3)
    assert not batch[properties.pbc].any()


def test_statistics_structures_from_a_dataset():
    structures = StatisticsStructures.from_dataset(DATASET)
    assert structures.n_atoms.tolist() == [0, 0, 1, 1, 1]
    assert structures.atom_types[[1, 6, 8]].tolist() == [5, 3, 1]


def test_statistics_structures_refuse_empty_structures():
    with pytest.raises(ValueError, match="zero atoms"):
        StatisticsStructures(torch.tensor([1, 1]), torch.tensor([0, 1]))


def test_the_prior_redraws_provided_structures_centered():
    torch.manual_seed(0)
    batch = GaussianPrior(10.0)(DatasetStructures(DATASET)(3))
    assert batch[properties.R].shape == (9, 3)
    for m in range(3):
        x = batch[properties.R][batch[properties.idx_m] == m]
        assert x.mean(0).norm().item() == pytest.approx(0.0, abs=1e-4)


def test_the_prior_draws_positions_for_statistics_structures():
    torch.manual_seed(0)
    structures = StatisticsStructures(torch.tensor([0, 0, 1]), torch.tensor([0, 1]))
    batch = GaussianPrior(1.0)(structures(4))
    assert batch[properties.R].shape == (8, 3)


def test_statistics_structures_without_types_leave_them_to_a_prior():
    torch.manual_seed(0)
    structures = StatisticsStructures.from_dataset(DATASET, atom_types=False)
    assert structures.atom_types is None
    batch = structures(5)
    assert properties.Z not in batch
    assert batch[properties.idx_m].shape == (int(batch[properties.n_atoms].sum()),)
    z = MaskPrior()(batch)[properties.Z]
    assert z.tolist() == [MASK] * int(batch[properties.n_atoms].sum())


def test_statistics_structures_can_be_periodic_and_leave_the_cell_to_a_prior():
    structures = StatisticsStructures.from_dataset(DATASET, atom_types=False, pbc=True)
    batch = structures(4)
    assert batch[properties.pbc].shape == (12,) and batch[properties.pbc].all()
    assert (batch[properties.cell] == 0).all()
    assert not StatisticsStructures.from_dataset(DATASET)(4)[properties.pbc].any()

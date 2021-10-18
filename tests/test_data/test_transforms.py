import schnetpack.properties as structure
import pytest
import torch
from ase.data import atomic_masses
from schnetpack.transform import *


def assert_consistent(orig, transformed):
    for k, v in orig.items():
        assert (v == transformed[k]).all(), f"Changed value: {k}"


@pytest.fixture(params=[0, 1])
def neighbor_list(request):
    neighbor_lists = [ASENeighborList, TorchNeighborList]
    return neighbor_lists[request.param]


class TestNeighborLists:
    """
    Test for different neighbor lists defined in neighbor_list using the Argon environment fixtures (periodic and
    non-periodic).

    """

    def test_neighbor_list(self, neighbor_list, environment):
        cutoff, props, neighbors_ref = environment
        neighbor_list = neighbor_list(cutoff)
        neighbors = neighbor_list(props)
        R = props[structure.R]
        neighbors[structure.Rij] = (
            R[neighbors[structure.idx_j]]
            - R[neighbors[structure.idx_i]]
            + props[structure.offsets]
        )

        neighbors = self._sort_neighbors(neighbors)
        neighbors_ref = self._sort_neighbors(neighbors_ref)

        for nbl, nbl_ref in zip(neighbors, neighbors_ref):
            torch.testing.assert_allclose(nbl, nbl_ref)

    def _sort_neighbors(self, neighbors):
        """
        Routine for sorting the index, shift and distance vectors to allow comparison between different
        neighbor list implementations.

        Args:
            neighbors: Input dictionary holding system neighbor information (idx_i, idx_j, cell_offset and Rij)

        Returns:
            torch.LongTensor: indices of central atoms in each pair
            torch.LongTensor: indices of each neighbor
            torch.LongTensor: cell offsets
            torch.Tensor: distance vectors associated with each pair
        """
        idx_i = neighbors[structure.idx_i]
        idx_j = neighbors[structure.idx_j]
        Rij = neighbors[structure.Rij]

        sort_idx = self._get_unique_idx(idx_i, idx_j, Rij)

        return idx_i[sort_idx], idx_j[sort_idx], Rij[sort_idx]

    @staticmethod
    def _get_unique_idx(
        idx_i: torch.Tensor, idx_j: torch.Tensor, offsets: torch.Tensor
    ):
        """
        Compute unique indices for every neighbor pair based on the central atom, the neighbor and the cell the
        neighbor belongs to. This is used for sorting the neighbor lists in order to compare between different
        implementations.

        Args:
            idx_i: indices of central atoms in each pair
            idx_j: indices of each neighbor
            offsets: cell offsets

        Returns:
            torch.LongTensor: indices used for sorting each tensor in a unique manner
        """
        n_max = torch.max(torch.abs(offsets))

        n_repeats = 2 * n_max + 1
        n_atoms = torch.max(idx_i) + 1

        unique_idx = (
            n_repeats ** 3 * (n_atoms * idx_i + idx_j)
            + (offsets[:, 0] + n_max)
            + n_repeats * (offsets[:, 1] + n_max)
            + n_repeats ** 2 * (offsets[:, 2] + n_max)
        )

        return torch.argsort(unique_idx)


def test_single_atom(single_atom, neighbor_list, cutoff):
    neighbor_list = neighbor_list(cutoff)
    props_after = neighbor_list(single_atom)
    R = props_after[structure.R]
    props_after[structure.Rij] = (
        R[props_after[structure.idx_j]]
        - R[props_after[structure.idx_i]]
        + props_after[structure.offsets]
    )

    assert_consistent(single_atom, props_after)
    assert len(props_after[structure.offsets]) == 0
    assert len(props_after[structure.idx_i]) == 0
    assert len(props_after[structure.idx_j]) == 0


def test_cast(single_atom):
    allf64 = [k for k, v in single_atom.items() if v.dtype is torch.float64]
    other_types = {
        k: v.dtype for k, v in single_atom.items() if v.dtype is not torch.float64
    }

    assert len(allf64) > 0, single_atom
    props_after = CastTo32()(single_atom)

    for k in props_after:
        if k in allf64:
            assert props_after[k].dtype is torch.float32
        else:
            assert props_after[k].dtype is other_types[k]


def test_remove_com(four_atoms):
    positions_trans = SubtractCenterOfMass()(four_atoms)

    com = torch.tensor([0.0, 0.0, 0.0])
    for r_i, m_i in zip(
        positions_trans[structure.position], atomic_masses[four_atoms[structure.Z]]
    ):
        com += r_i * m_i

    torch.testing.assert_allclose(com, torch.tensor([0.0, 0.0, 0.0]))


def test_remove_cog(four_atoms):
    positions_trans = SubtractCenterOfGeometry()(four_atoms)

    cog = torch.tensor([0.0, 0.0, 0.0])
    for r_i in positions_trans[structure.position]:
        cog += r_i

    torch.testing.assert_allclose(cog, torch.tensor([0.0, 0.0, 0.0]))

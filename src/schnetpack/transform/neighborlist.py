import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from typing import Dict, Optional
from .transform import Transform

__all__ = [
    "NeighborListTransform",
    "ASENeighborList",
    "TorchNeighborList",
    "CollectAtomTriples",
    "CountNeighbors",
]

from schnetpack import properties as structure


class NeighborListTransform(Transform):
    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, cutoff: float, return_offset: bool = False):
        super(NeighborListTransform, self).__init__()

        self.cutoff = cutoff
        self.return_offset = return_offset

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
        wrap: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        return self._build_neighbor_list(inputs)

    def _build_neighbor_list(self, inputs: Dict[str, torch.Tensor]):
        raise NotImplementedError


class ASENeighborList(NeighborListTransform):
    def __init__(self, cutoff, return_offset=False):
        super(ASENeighborList, self).__init__(
            cutoff=cutoff, return_offset=return_offset
        )

    def _build_neighbor_list(self, inputs: Dict[str, torch.Tensor]):
        Z = inputs[structure.Z]
        R = inputs[structure.R]
        cell = inputs[structure.cell]
        pbc = inputs[structure.pbc]
        at = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)

        if self.return_offset:
            idx_i, idx_j, Rij, S = neighbor_list(
                "ijDS", at, self.cutoff, self_interaction=False
            )
            inputs[structure.offsets] = torch.tensor(S)
        else:
            idx_i, idx_j, Rij = neighbor_list(
                "ijD", at, self.cutoff, self_interaction=False
            )

        inputs[structure.idx_i] = torch.tensor(idx_i)
        inputs[structure.idx_j] = torch.tensor(idx_j)
        inputs[structure.Rij] = torch.tensor(Rij)
        return inputs


class TorchNeighborList(NeighborListTransform):
    def __init__(self, cutoff, return_offset=False):
        super(TorchNeighborList, self).__init__(
            cutoff=cutoff, return_offset=return_offset
        )

    def _build_neighbor_list(self, inputs: Dict[str, torch.Tensor]):
        positions = inputs[structure.R]
        pbc = inputs[structure.pbc]
        cell = inputs[structure.cell]

        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device).long()
        else:
            shifts = self._get_shifts(cell, pbc)

        idx_i, idx_j, Rij, offsets = self._get_neighbor_pairs(positions, cell, shifts)

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((idx_i, idx_j), dim=0)
        bi_idx_j = torch.cat((idx_j, idx_i), dim=0)
        bi_Rij = torch.cat((-Rij, Rij), dim=0)

        # Sort along first dimension (necessary for atom-wise pooling)
        sorted_idx = torch.argsort(bi_idx_i)

        inputs[structure.idx_i] = bi_idx_i[sorted_idx]
        inputs[structure.idx_j] = bi_idx_j[sorted_idx]
        inputs[structure.Rij] = bi_Rij[sorted_idx]

        if self.return_offset:
            bi_offset = torch.cat((-offsets, offsets), dim=0)
            inputs[structure.offsets] = bi_offset[sorted_idx]

        return inputs

    def _get_neighbor_pairs(self, positions, cell, shifts):
        """Compute pairs of atoms that are neighbors
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # 1) Central cell
        pi_center, pj_center = torch.combinations(all_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, all_atoms, all_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoff
        distances = torch.norm(Rij_all, dim=1)
        in_cutoff = torch.nonzero(distances < self.cutoff, as_tuple=False)

        # 6) Reduce tensors to relevant components
        pair_index = in_cutoff.squeeze()
        atom_index_i = pi_all[pair_index]
        atom_index_j = pj_all[pair_index]
        Rij = Rij_all.index_select(0, pair_index)
        offsets = shifts_all[pair_index]

        return atom_index_i, atom_index_j, Rij, offsets

    def _get_shifts(self, cell, pbc):
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.

        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(self.cutoff * inverse_lengths).long()
        num_repeats = torch.where(
            pbc, num_repeats, torch.Tensor([0], device=cell.device).long()
        )

        r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)

        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )


class CollectAtomTriples(Transform):
    """
    Generate the index tensors for all triples between atoms within the cutoff shell.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Using the neighbors contained within the cutoff shell, generate all unique pairs of neighbors and convert
        them to index arrays. Applied to the neighbor arrays, these arrays generate the indices involved in the atom
        triples.

        E.g.:
            idx_j[idx_j_triples] -> j atom in triple
            idx_j[idx_k_triples] -> k atom in triple
            Rij[idx_j_triples] -> Rij vector in triple
            Rij[idx_k_triples] -> Rik vector in triple
        """
        idx_i = inputs[structure.idx_i]

        _, n_neighbors = torch.unique_consecutive(idx_i, return_counts=True)

        offset = 0
        idx_i_triples = ()
        idx_jk_triples = ()
        for idx in range(n_neighbors.shape[0]):
            triples = torch.combinations(
                torch.arange(offset, offset + n_neighbors[idx]), r=2
            )
            idx_i_triples += (torch.ones(triples.shape[0], dtype=torch.long) * idx,)
            idx_jk_triples += (triples,)
            offset += n_neighbors[idx]

        idx_i_triples = torch.cat(idx_i_triples)

        idx_jk_triples = torch.cat(idx_jk_triples)
        idx_j_triples, idx_k_triples = idx_jk_triples.split(1, dim=-1)

        inputs[structure.idx_i_triples] = idx_i_triples
        inputs[structure.idx_j_triples] = idx_j_triples.squeeze(-1)
        inputs[structure.idx_k_triples] = idx_k_triples.squeeze(-1)

        return inputs


class CountNeighbors(Transform):
    """
    Store the number of neighbors for each atom
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, sorted: bool = True):
        """
        Args:
            sorted: Set to false if chosen neighbor list yields unsorted center indices (idx_i).
        """
        super(CountNeighbors, self).__init__()
        self.sorted = sorted

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        idx_i = inputs[structure.idx_i]

        if self.sorted:
            _, n_nbh = torch.unique_consecutive(idx_i, return_counts=True)
        else:
            _, n_nbh = torch.unique(idx_i, return_counts=True)

        inputs[structure.n_nbh] = n_nbh
        return inputs

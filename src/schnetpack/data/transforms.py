from typing import Dict, Optional

import torch
import torch.nn as nn
import schnetpack.structure as structure
import schnetpack as spk

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import atomic_masses

__all__ = [
    "ASENeighborList",
    "TorchNeighborList",
    "CastMap",
    "CastTo32",
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
]

## neighbor lists
class TransformException(Exception):
    pass


class Transform(nn.Module):
    """
    Base class for all transforms

    Currently, this only ensures that the reference to the data attributeis initialized.
    """

    data: Optional["spk.data.BaseAtomsData"]

    def __init__(self):
        self.data = None
        super().__init__()


class ASENeighborList(Transform):
    """
    Calculate neighbor list using ASE.

    Note: This is quite slow and should only used as a baseline for faster implementations!
    """

    def __init__(self, cutoff):
        """
        Args:
            cutoff: cutoff radius for neighbor search
        """
        super().__init__()
        self.cutoff = cutoff

    def forward(self, inputs):
        Z = inputs[structure.Z]
        R = inputs[structure.R]
        cell = inputs[structure.cell]
        pbc = inputs[structure.pbc]
        at = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
        idx_i, idx_j, idx_S, Rij = neighbor_list(
            "ijSD", at, self.cutoff, self_interaction=False
        )
        inputs[structure.idx_i] = torch.tensor(idx_i)
        inputs[structure.idx_j] = torch.tensor(idx_j)
        inputs[structure.Rij] = torch.tensor(Rij)
        inputs[structure.cell_offset] = torch.tensor(idx_S)
        return inputs


class TorchNeighborList(Transform):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).
    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    Args:
        cutoff: cutoff radius for neighbor search
    """

    def __init__(self, cutoff):
        super(TorchNeighborList, self).__init__()
        self.cutoff = cutoff

    def forward(self, inputs):
        positions = inputs[structure.R]
        pbc = inputs[structure.pbc]
        cell = inputs[structure.cell]

        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device).long()
        else:
            shifts = self._get_shifts(cell, pbc)

        idx_i, idx_j, idx_S, Rij = self._get_neighbor_pairs(positions, cell, shifts)

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((idx_i, idx_j), dim=0)
        bi_idx_j = torch.cat((idx_j, idx_i), dim=0)
        bi_idx_S = torch.cat((-idx_S, idx_S), dim=0)
        bi_Rij = torch.cat((-Rij, Rij), dim=0)

        # Sort along first dimension (necessary for atom-wise pooling)
        sorted_idx = torch.argsort(bi_idx_i)

        inputs[structure.idx_i] = bi_idx_i[sorted_idx]
        inputs[structure.idx_j] = bi_idx_j[sorted_idx]
        inputs[structure.Rij] = bi_Rij[sorted_idx]
        inputs[structure.cell_offset] = bi_idx_S[sorted_idx]

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
        shifts = shifts_all.index_select(0, pair_index)
        Rij = Rij_all.index_select(0, pair_index)

        return atom_index_i, atom_index_j, shifts, Rij

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


## casting


class CastMap(Transform):
    """
    Cast all inputs according to type map.
    """

    def __init__(self, type_map: Dict[torch.dtype, torch.dtype]):
        """
        Args:
            type_map: dict with soource_type: target_type
        """
        super().__init__()
        self.type_map = type_map

    def forward(self, inputs):
        for k, v in inputs.items():
            if v.dtype in self.type_map:
                inputs[k] = v.to(dtype=self.type_map[v.dtype])
        return inputs


class CastTo32(CastMap):
    """ Cast all float64 tensors to float32 """

    def __init__(self):
        super().__init__(type_map={torch.float64: torch.float32})


## centering


class SubtractCenterOfMass(Transform):
    """
    Subtract center of mass from positions. Can only be used for single structures. Batches of structures are not supported.

    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        masses = torch.tensor(atomic_masses[inputs[structure.Z]])
        inputs[structure.position] -= (
            masses.unsqueeze(-1) * inputs[structure.position]
        ).sum(0) / masses.sum()
        return inputs


class SubtractCenterOfGeometry(Transform):
    """
    Subtract center of geometry from positions. Can only be used for single structures. Batches of structures are not supported.

    """

    def forward(self, inputs):
        inputs[structure.position] -= inputs[structure.position].mean(0)
        return inputs


class UnitConversion(Transform):
    def __init__(self, property_unit_map: Dict[str, str]):
        self.property_unit_map = property_unit_map
        self.src_units = None
        super().__init__()

    def forward(self, inputs):
        # initialize
        if not self.src_units:
            units = self.data.units
            self.src_units = {p: units[p] for p in self.property_unit_map}

        for prop, tgt_unit in self.property_unit_map.items():
            inputs[prop] *= spk.units.convert_units(self.src_units[prop], tgt_unit)
        return inputs

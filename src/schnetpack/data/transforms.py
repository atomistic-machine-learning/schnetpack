from typing import Dict

import torch
import torch.nn as nn
from schnetpack import Structure

from ase import Atoms
from ase.neighborlist import neighbor_list


## neighbor lists


class ASENeighborList(nn.Module):
    """
    Calculate neighbor list using ASE.

    Note: This is quite slow and should only used as a baseline for faster implementations!

    Args:
        cutoff: cutoff radius for neighbor search
    """

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, inputs):
        Z = inputs[Structure.Z]
        R = inputs[Structure.R]
        cell = inputs[Structure.cell]
        pbc = inputs[Structure.pbc]
        at = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
        idx_i, idx_j, idx_S, Rij = neighbor_list(
            "ijSD", at, self.cutoff, self_interaction=False
        )
        inputs[Structure.idx_i] = torch.tensor(idx_i)
        inputs[Structure.idx_j] = torch.tensor(idx_j)
        inputs[Structure.Rij] = torch.tensor(Rij)
        inputs[Structure.cell_offset] = torch.tensor(idx_S)
        return inputs


## casting


class CastMap(nn.Module):
    """
    Cast all inputs according to type map.

    Args:
        type_map: dict with soource_type: target_type
    """

    def __init__(self, type_map: Dict[torch.dtype, torch.dtype]):
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

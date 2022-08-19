from typing import Dict, Optional

import torch
import torch.nn as nn

import schnetpack.properties as properties


class PairwiseDistances(nn.Module):
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        R = inputs[properties.R]
        offsets = inputs[properties.offsets]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        Rij = R[idx_j] - R[idx_i] + offsets
        inputs[properties.Rij] = Rij
        return inputs


class FilterShortRange(nn.Module):
    """
    Separate short-range from all supplied distances.

    The short-range distances will be stored under the original keys (properties.Rij,
    properties.idx_i, properties.idx_j), while the original distances can be accessed for long-range terms via
    (properties.Rij_lr, properties.idx_i_lr, properties.idx_j_lr).
    """

    def __init__(self, short_range_cutoff: float):
        super().__init__()
        self.short_range_cutoff = short_range_cutoff

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        Rij = inputs[properties.Rij]

        rij = torch.norm(Rij, dim=-1)
        cidx = torch.nonzero(rij <= self.short_range_cutoff).squeeze(-1)

        inputs[properties.Rij_lr] = Rij
        inputs[properties.idx_i_lr] = idx_i
        inputs[properties.idx_j_lr] = idx_j

        inputs[properties.Rij] = Rij[cidx]
        inputs[properties.idx_i] = idx_i[cidx]
        inputs[properties.idx_j] = idx_j[cidx]
        return inputs

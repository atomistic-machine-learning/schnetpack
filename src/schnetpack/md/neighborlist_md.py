from typing import Dict

import torch

from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import (
    BatchNeighborList,
    CollectAtomTriples,
    NeighborListTransform,
)

__all__ = ["NeighborListMD"]


class NeighborListMD:
    """
    Wrapper for neighbor list transforms to make them suitable for molecular dynamics simulations. Introduces handling
    of multiple replicas and a cutoff shell (buffer region) to avoid recomputations of the neighbor list in every step.

    The work is done by :class:`~schnetpack.transform.BatchNeighborList`, shared with batchwise structure relaxation
    framework.
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_shell: float,
        base_nbl: NeighborListTransform,
        requires_triples: bool = False,
        collate_fn: callable = _atoms_collate_fn,
    ):
        """

        Args:
            cutoff (float): Cutoff radius.
            cutoff_shell (float): Buffer region. Atoms can move this much unitil neighbor list needs to be recomputed.
            base_nbl (schnetpack.transform.NeighborListTransform): basic SchNetPack neighbor list transform.
            requires_triples (bool): Compute atom triples, e.g. for angles (default=False).
            collate_fn (callable): Collate function for batch generation. Used to combine neighbor lists of differnt
                                   replicas and molecules.
        """
        self.cutoff = cutoff
        self.cutoff_shell = cutoff_shell
        self.cutoff_full = cutoff + cutoff_shell
        self.requires_triples = requires_triples
        self._collate = collate_fn

        self.neighbor_list = BatchNeighborList(
            neighbor_list=base_nbl(cutoff),
            cutoff_skin=cutoff_shell,
            transforms=[CollectAtomTriples()] if requires_triples else None,
        )

    def get_neighbors(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute neighbor indices from positions and simulations cells.

        Args:
            inputs (dict(str, torch.Tensor)): input batch.

        Returns:
            dict(str, torch.Tensor): indices of neighbors, and nothing else -- the caller
            merges them into the batch it already holds.
        """
        return self.neighbor_list.neighbors(inputs)

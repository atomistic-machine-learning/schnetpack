"""Neighbor lists for a batch of structures that keeps moving.

A relaxation or an MD run evaluates the same structures over and over, moving them a
little each step. Rebuilding their neighbor lists every time is wasteful, so
:class:`BatchNeighborList` builds them out to ``cutoff + cutoff_skin`` and reuses them for
as long as no atom of a structure has drifted more than half the skin. Each step the
cached list is restricted to the pairs actually within the cutoff, which is a single
masked gather on the device the batch already lives on.

This is the batch-level counterpart of
:class:`~schnetpack.transform.SkinNeighborList`, which does the same thing one sample at a
time while a batch is being assembled. Unlike the transforms it is not a
:class:`~schnetpack.transform.Transform`: it holds state, and it works on a collated batch
rather than on a single sample.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn, split_batch

from .base import Transform

__all__ = ["BatchNeighborList"]

#: entries of a cached list that are pair-indexed, i.e. shrink when the skin is pruned
_PAIR_KEYS = (
    properties.idx_i,
    properties.idx_j,
    properties.offsets,
    properties.lidx_i,
    properties.lidx_j,
)

#: entries that index into the pair arrays and have to be renumbered along with them
_TRIPLE_KEYS = (properties.idx_j_triples, properties.idx_k_triples)


class BatchNeighborList:
    """Keeps the neighbor lists of a batch valid while its structures move.

    The batch is a schnetpack input dictionary and stays on its device throughout, except
    on the steps that have to rebuild: the neighbor list implementations run on cpu, so
    the structures needing a new list -- and only those -- are shipped over and back.

    ``idx_i``, ``idx_j`` and ``offsets`` are what this produces; the ``Rij`` that can come
    with them is a by-product of the pruning, and models recompute it with their
    ``PairwiseDistances`` input module anyway.

    Note:
        The offsets of a pair list are cell shifts, so pruning takes ``R[j] - R[i] +
        offsets`` for the pair vector -- the convention ``PairwiseDistances`` and every
        neighbor list in schnetpack share. Atoms that have wandered outside their cell
        break it, here and in the model alike, so a structure with a cell has to stay in
        it. Structures without one are unaffected.

    Args:
        neighbor_list: the neighbor list transform to build with, e.g.
            :class:`~schnetpack.transform.MatScipyNeighborList`. Its cutoff is read off and
            then widened by ``cutoff_skin`` -- the transform is modified in place, the way
            :class:`~schnetpack.transform.SkinNeighborList` does it.
        cutoff_skin: an atom may drift half of this before its structure's list is
            rebuilt. A wider skin means fewer rebuilds and more pairs to carry along.
        transforms: transforms applied to each structure after its neighbor list is built,
            e.g. :class:`~schnetpack.transform.CollectAtomTriples`.
        device: device the returned entries live on. Defaults to following the batch,
            which is what a caller propagating structures on a device wants.
        dtype: float precision of the returned entries. Defaults to following the batch.
        additional_inputs: entries added to every structure before the transforms run, for
            transforms that need them.
    """

    def __init__(
        self,
        neighbor_list: Transform,
        cutoff_skin: float = 0.3,
        transforms: Optional[Union[Transform, List[Transform]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        additional_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype
        self.additional_inputs = additional_inputs or {}

        self.cutoff = neighbor_list._cutoff
        self.cutoff_skin = cutoff_skin
        # build out to cutoff + skin, so a list stays usable while the atoms move
        neighbor_list._cutoff = self.cutoff + cutoff_skin

        if transforms is None:
            transforms = []
        elif not isinstance(transforms, list):
            transforms = [transforms]

        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError(f"Unrecognized precision {dtype}")

        self.transforms: List[Transform] = [neighbor_list] + transforms

        # resolved per call, from the batch, unless they were pinned in the constructor
        self.device = self._device or torch.device("cpu")
        self.dtype = self._dtype or torch.float32

        self.reset()

    def reset(self) -> None:
        """Forget every cached list, so that the next call rebuilds from scratch."""
        #: the cutoff+skin list of each structure, with the structure it was built for
        self._references: Dict[int, Dict[str, torch.Tensor]] = {}
        #: those lists concatenated into batch numbering, on device, ready to be pruned
        self._cache: Optional[Dict[str, torch.Tensor]] = None

    # -------------------------------------------------------------- public interface

    def update(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The batch with its neighbor lists refreshed for the current positions.

        Entries the caller put in the batch are carried over untouched; only the
        neighborhood entries are replaced.
        """
        return {**inputs, **self.neighbors(inputs, with_distances=True)}

    def neighbors(
        self, inputs: Dict[str, torch.Tensor], with_distances: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Just the neighborhood entries for the structures in ``inputs``.

        Deliberately free of positions, atomic numbers, cells and atom counts, so that a
        caller holding the live batch can ``update()`` its dictionary with the result
        without overwriting the very tensors it is propagating.

        Args:
            inputs: input batch. Only the structure-defining entries are read.
            with_distances: also return ``Rij`` for the surviving pairs.
        """
        positions = inputs[properties.R]
        self.device = self._device or positions.device
        self.dtype = self._dtype or (
            positions.dtype if positions.is_floating_point() else torch.float32
        )

        stale = self._stale_structures(inputs)
        if stale:
            self._rebuild(inputs, stale)
        return self._prune(inputs, with_distances=with_distances)

    # -------------------------------------------------------------- rebuild or reuse

    def _stale_structures(self, inputs: Dict[str, torch.Tensor]) -> List[int]:
        """Which structures have moved far enough to need a new list.

        Answered per structure and on the device the batch already lives on, against the
        positions each structure's own list was built for. A structure drifting just under
        the threshold must not have its budget reset because another structure in the
        batch was rebuilt -- which is what measuring against one batch-wide reference,
        refreshed whenever anything rebuilds, would do.
        """
        n_atoms = inputs[properties.n_atoms]
        n_structures = int(n_atoms.shape[0])

        if self._cache is None or len(self._references) != n_structures:
            # a batch of a different size is a different batch; nothing may be reused
            self.reset()
            return list(range(n_structures))

        positions = inputs[properties.R]
        if positions.shape[0] != self._cache[properties.R].shape[0]:
            self.reset()
            return list(range(n_structures))

        squared_drift = (
            (self._cache[properties.R] - positions).pow(2).sum(-1).to(torch.float64)
        )
        idx_m = torch.repeat_interleave(
            torch.arange(n_structures, device=positions.device), n_atoms
        )
        per_structure = torch.zeros(
            n_structures, dtype=torch.float64, device=positions.device
        ).scatter_reduce_(0, idx_m, squared_drift, reduce="amax", include_self=False)

        stale = per_structure >= 0.25 * self.cutoff_skin**2
        stale |= ~(
            torch.isclose(
                self._cache[properties.cell],
                inputs[properties.cell].view(n_structures, 3, 3).to(self.dtype),
            )
            .view(n_structures, -1)
            .all(-1)
        )
        stale |= (
            self._cache[properties.pbc] != inputs[properties.pbc].view(n_structures, 3)
        ).any(-1)

        return torch.nonzero(stale).view(-1).tolist()

    def _rebuild(self, inputs: Dict[str, torch.Tensor], stale: Sequence[int]) -> None:
        """Build fresh cutoff+skin lists for the given structures, and cache them.

        Straight off the batch: the structures are cut out of it as tensors and handed to
        the neighbor list transforms, which read positions, atomic numbers, cells and pbc
        out of an input dictionary and never need ``ase.Atoms``.
        """
        samples = split_batch(inputs)

        for idx in stale:
            sample = {key: value.cpu() for key, value in samples[idx].items()}
            sample.update(self.additional_inputs)
            for transform in self.transforms:
                sample = transform(sample)

            # the positions a list was built for are the reference the drift check
            # measures against, so they are kept alongside it. Kept on cpu, where they
            # were built and where they are concatenated; only the concatenated result
            # is worth moving to the device.
            self._references[idx] = {
                key: value for key, value in sample.items() if key != properties.idx
            }

        self._cache = self._collate_references(len(samples))

    def _collate_references(self, n_structures: int) -> Dict[str, torch.Tensor]:
        """Concatenate the per structure lists into batch numbering, once per rebuild.

        The pair indices of a structure count from its own first atom, so they have to be
        shifted by everything ahead of it -- which is exactly what the collate function
        the data loader uses does, triples included. It runs on cpu, where the lists were
        built; what lands on the device is the finished, concatenated cache, which every
        step from here to the next rebuild reads without touching the cpu again.
        """
        collated = _atoms_collate_fn(
            [self._references[idx] for idx in range(n_structures)]
        )
        collated = {key: value.to(self.device) for key, value in collated.items()}
        collated[properties.R] = collated[properties.R].to(self.dtype)
        collated[properties.offsets] = collated[properties.offsets].to(self.dtype)
        collated[properties.cell] = collated[properties.cell].view(n_structures, 3, 3)
        collated[properties.pbc] = collated[properties.pbc].view(n_structures, 3)
        return collated

    def _prune(
        self, inputs: Dict[str, torch.Tensor], with_distances: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Restrict the cached cutoff+skin lists to the pairs within the cutoff.

        The whole batch at once and on its own device -- the counterpart of
        :meth:`SkinNeighborList._remove_neighbors_in_skin`, which does the same thing one
        structure at a time on the cpu.
        """
        cache = self._cache
        idx_i, idx_j = cache[properties.idx_i], cache[properties.idx_j]
        offsets = cache[properties.offsets]

        positions = inputs[properties.R]
        Rij = positions[idx_j] - positions[idx_i] + offsets
        within_cutoff = Rij.pow(2).sum(-1) <= self.cutoff**2

        neighbors = {
            key: cache[key][within_cutoff] for key in _PAIR_KEYS if key in cache
        }
        if with_distances:
            neighbors[properties.Rij] = Rij[within_cutoff]

        if properties.idx_i_triples in cache:
            neighbors.update(self._prune_triples(cache, within_cutoff))

        return neighbors

    @staticmethod
    def _prune_triples(
        cache: Dict[str, torch.Tensor], within_cutoff: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Renumber the triples onto the pairs that survived the pruning.

        ``idx_j_triples`` and ``idx_k_triples`` index into the pair arrays, so dropping
        pairs without renumbering would leave them pointing at the wrong pairs, or past
        the end of the array altogether. Triples with a leg that did not survive are
        dropped.
        """
        renumbered = torch.full(
            within_cutoff.shape,
            -1,
            dtype=torch.long,
            device=within_cutoff.device,
        )
        renumbered[within_cutoff] = torch.arange(
            int(within_cutoff.sum()), device=within_cutoff.device
        )

        legs = [cache[key] for key in _TRIPLE_KEYS]
        keep = renumbered[legs[0]] >= 0
        for leg in legs[1:]:
            keep &= renumbered[leg] >= 0

        triples = {properties.idx_i_triples: cache[properties.idx_i_triples][keep]}
        for key, leg in zip(_TRIPLE_KEYS, legs):
            triples[key] = renumbered[leg[keep]]
        return triples

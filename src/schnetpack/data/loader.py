"""
The loader of atomistic datasets, :class:`AtomsLoader`: torch's ``DataLoader`` that
reads a dataset structure by structure, or slices whole padded batches from an
:class:`~schnetpack.data.AtomsCache`, and then moves each batch to a device and
passes it through the batch transforms.
"""

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _T_co

import schnetpack.properties as structure
from schnetpack.data.sampler import BudgetBatches, OrderedBatches
from schnetpack.transform.base import Transform

__all__ = ["AtomsLoader", "prune_neighbors"]


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
    )
    coll_batch[structure.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for _idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


#: entries of a neighbor list that are pair-indexed, i.e. shrink when it is pruned
_PAIR_KEYS = (
    structure.idx_i,
    structure.idx_j,
    structure.offsets,
    structure.lidx_i,
    structure.lidx_j,
)

#: entries that index into the pair arrays and have to be renumbered along with them
_TRIPLE_KEYS = (structure.idx_j_triples, structure.idx_k_triples)


def prune_neighbors(
    neighbors: dict[str, torch.Tensor],
    positions: torch.Tensor,
    cutoff: float,
    with_distances: bool = False,
) -> dict[str, torch.Tensor]:
    """The entries of a neighbor list, restricted to the pairs within ``cutoff``.

    A pair is kept when ``|R[j] - R[i] + offsets| <= cutoff``. The pair-indexed
    entries are pruned together, on the device they live on, and atom triples are
    renumbered onto the kept pairs. Only these entries come back, so the caller
    merges them into its batch.

    Args:
        neighbors: the neighbor list, ``idx_i``, ``idx_j`` and ``offsets``, with
            any other pair-indexed entries and triples to prune along with them.
        positions: the positions the pairs are measured at, in the length unit of
            ``offsets`` and ``cutoff``.
        cutoff: the cutoff to prune to.
        with_distances: also return ``Rij`` of the kept pairs.
    """
    idx_i, idx_j = neighbors[structure.idx_i], neighbors[structure.idx_j]
    Rij = positions[idx_j] - positions[idx_i] + neighbors[structure.offsets]
    within_cutoff = Rij.pow(2).sum(-1) <= cutoff**2

    pruned = {
        key: neighbors[key][within_cutoff] for key in _PAIR_KEYS if key in neighbors
    }
    if with_distances:
        pruned[structure.Rij] = Rij[within_cutoff]

    if structure.idx_i_triples in neighbors:
        pruned.update(_prune_triples(neighbors, within_cutoff))

    return pruned


def _prune_triples(
    neighbors: dict[str, torch.Tensor], within_cutoff: torch.Tensor
) -> dict[str, torch.Tensor]:
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

    legs = [neighbors[key] for key in _TRIPLE_KEYS]
    keep = renumbered[legs[0]] >= 0
    for leg in legs[1:]:
        keep &= renumbered[leg] >= 0

    triples = {structure.idx_i_triples: neighbors[structure.idx_i_triples][keep]}
    for key, leg in zip(_TRIPLE_KEYS, legs):
        triples[key] = renumbered[leg[keep]]
    return triples


class _AsArrays:
    """A collate function whose batches leave the worker as numpy arrays."""

    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, samples):
        batch = self.collate_fn(samples)
        return {
            k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }


def _as_tensors(batches):
    for batch in batches:
        yield {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in batch.items()
        }


class _Positions(Dataset):
    """The dataset torch's DataLoader sees with a cache: the batch sampler draws
    positions in the cache, and the collate function gathers them."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitems__(self, positions):
        return positions


class _CachedBatch:
    """
    The collate function with a cache: the batch of some positions, padded to fixed
    sizes (:meth:`~schnetpack.data.AtomsCache.gather`), on ``device``. On a GPU it
    is gathered into pinned buffers that every batch reuses, and copied without
    waiting for the GPU, which may still work on the last step.
    """

    def __init__(self, cache, n_atoms_pad, n_pairs_pad, n_slots, device):
        self.cache, self.device = cache, device
        self.sizes = (n_atoms_pad, n_pairs_pad, n_slots)
        #: the pinned buffers and the event of their last copy
        self._pinned, self._copied = {}, None

    def __call__(self, positions) -> dict[str, torch.Tensor]:
        if self.device is not None and self.device.type == "cuda":
            if self._copied is not None:
                self._copied.synchronize()  # the last copy out of the buffers is done
            pinned = self.cache.gather(positions, *self.sizes, self._pinned)
            for key, value in pinned.items():
                if self._pinned.get(key) is not value:  # a new buffer
                    pinned[key] = self._pinned[key] = value.pin_memory()
            batch = {k: v.to(self.device, non_blocking=True) for k, v in pinned.items()}
            self._copied = torch.cuda.Event()
            self._copied.record(torch.cuda.current_stream(self.device))
            return batch
        return self.cache.gather(positions, *self.sizes)  # the loader moves it


def _default_generator(device: torch.device | None) -> torch.Generator:
    if device is not None and device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return torch.cuda.default_generators[index]
    return torch.default_generator


class AtomsLoader(DataLoader):
    """
    Batches of an atomistic dataset (e.g. a view of
    :class:`~schnetpack.data.ASEAtomsData`), on torch's ``DataLoader``.

    Without a cache, structures are read one by one with the dataset's transforms,
    in worker processes if asked, and collated. With worker processes (and no
    pinned memory), a collated batch crosses to the main process as numpy arrays,
    one pickled message per batch, and becomes tensors there without a copy.
    Tensors would cross through shared memory one file descriptor at a time, a
    socket round trip each: receiving a QM9 batch of 13 tensors cost the main
    process 6.5 ms, a third of a compiled training step (16 workers, H100 PCIe).

    With a cache (:class:`~schnetpack.data.AtomsCache`), the dataset only names the
    structures, and whole batches are sliced from the cache in the main process,
    padded to fixed sizes (``properties.atom_mask`` and
    ``properties.structure_mask`` mark the real rows). In order (validation,
    testing), every structure comes once and all batches are padded to the largest
    one. Shuffled (training), the batches are packed to the atoms and pairs that
    ``batch_size`` structures hold on average
    (:class:`~schnetpack.data.BudgetBatches`): every batch has the same shapes and
    little of it is padding.

    On both routes each batch is then moved to ``device`` and passed through
    ``batch_transforms``, which act on collated batches (offsets, casts, noise, a
    neighbor list after the noise). ::

        train = AtomsLoader(data.subset(train_idx), batch_size=100, shuffle=True,
                            num_workers=8, batch_transforms=[CastTo32()], device="cuda")
        cache = AtomsCache.from_dataset(data)
        train = AtomsLoader(data.subset(train_idx), batch_size=100, shuffle=True,
                            cache=cache, batch_transforms=[CastTo32()], device="cuda")
    """

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: int | None = 1,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        batch_sampler: Sampler[list[int]] | Iterable[list[int]] | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t = _atoms_collate_fn,
        pin_memory: bool = False,
        cache=None,
        batch_transforms: list[Transform] | None = None,
        device: str | torch.device | None = None,
        seed: int = 0,
        transform_seed: int | None = None,
        **kwargs,
    ):
        """
        Args:
            dataset: the structures; with a cache, a view of the dataset the cache
                was made from
            batch_size: structures per batch (shuffled with a cache: on average)
            shuffle: a new order every pass
            sampler, batch_sampler, num_workers, collate_fn, pin_memory: as for
                torch's ``DataLoader``; not with a cache
            cache: an :class:`~schnetpack.data.AtomsCache` that holds the
                structures of ``dataset``; whole padded batches are sliced from it
            batch_transforms: transforms of every collated batch, on ``device``
            device: where the batches go and the batch transforms run. None: as
                they come
            seed: with a cache and ``shuffle``, the seed of the orders
            transform_seed: random batch transforms (noise) draw the same values in
                every pass, from a random state of their own started from this
                seed. None: from the device's default generator
        """
        # each new argument as an attribute of its name: Lightning rebuilds a
        # DataLoader subclass from its attributes (to inject a sampler)
        self.cache = cache
        self.batch_transforms = batch_transforms
        self.device = None if device is None else torch.device(device)
        self.seed, self.transform_seed = seed, transform_seed
        self._as_arrays = cache is None and num_workers > 0 and not pin_memory
        if cache is None:
            super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=_AsArrays(collate_fn) if self._as_arrays else collate_fn,
                pin_memory=pin_memory,
                **kwargs,
            )
            return

        given = {
            "num_workers": num_workers != 0,
            "sampler": sampler is not None,
            "batch_sampler": batch_sampler is not None,
            "collate_fn": collate_fn is not _atoms_collate_fn,
            "pin_memory": pin_memory,
        }
        if any(given.values()):
            names = ", ".join(name for name, set_ in given.items() if set_)
            raise ValueError(
                f"with a cache, AtomsLoader takes no {names}: it slices whole "
                "batches from the cache in the main process"
            )
        # the batches are planned on numpy views of the sizes (the samplers draw
        # their orders from a numpy generator)
        positions = cache.positions(dataset)
        atoms, pairs = cache.n_atoms.numpy(), cache.n_pairs.numpy()
        if shuffle:
            batches = BudgetBatches(atoms, pairs, positions, batch_size, seed)
            # atoms (one padding atom at least) and pairs of every batch
            pad = (batches.atom_budget + 1, batches.pair_budget)
            slots = batches.max_structures
        else:
            batches = OrderedBatches(positions, batch_size)
            sizes = [(atoms[b].sum(), pairs[b].sum()) for b in batches]
            most = np.max(sizes, axis=0) if sizes else (0, 0)
            pad, slots = (int(most[0]) + 1, int(most[1])), None
        super().__init__(
            _Positions(len(cache)),
            batch_sampler=batches,
            collate_fn=_CachedBatch(cache, *pad, slots, self.device),
            # of its own: DataLoader.__iter__ draws a seed from the generator on
            # every pass, which would advance the global one
            generator=torch.Generator(),
            **kwargs,
        )

    def __iter__(self):
        batches = super().__iter__()
        if self._as_arrays:
            batches = _as_tensors(batches)
        return self._moved_and_transformed(batches)

    def _moved_and_transformed(self, batches):
        transforms, device = self.batch_transforms or [], self.device
        if device is not None:
            for t in transforms:  # at every pass: initialize may put buffers back
                t.to(device)
        if self.transform_seed is None:
            for batch in batches:
                yield self._transform(batch)
            return
        # the default generator is switched to the pass's own state while the
        # transforms run and back right after: everything else continues its own
        # sequence
        gen = _default_generator(device)
        outer = gen.get_state()
        own = gen.manual_seed(self.transform_seed).get_state()
        gen.set_state(outer)
        for batch in batches:
            outer = gen.get_state()
            gen.set_state(own)
            try:
                batch = self._transform(batch)
                own = gen.get_state()
            finally:
                gen.set_state(outer)
            yield batch

    def _transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.device is not None:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        for t in self.batch_transforms or []:
            batch = t(batch)
        return batch

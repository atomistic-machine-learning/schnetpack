import torch
from torch.utils.data import DataLoader

from typing import Dict, List, Optional, Sequence
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _T_co

import schnetpack.properties as structure

__all__ = ["AtomsLoader", "split_batch"]


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
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


#: entries that describe the structures themselves, as opposed to their neighborhoods.
#: These are what a neighbor list needs, and what :func:`split_batch` splits by default.
_STRUCTURE_KEYS = (
    structure.n_atoms,
    structure.Z,
    structure.R,
    structure.cell,
    structure.pbc,
)


def split_batch(
    inputs: Dict[str, torch.Tensor], keys: Optional[Sequence[str]] = None
) -> List[Dict[str, torch.Tensor]]:
    """Split a collated batch back into one input dictionary per structure.

    The inverse of :func:`_atoms_collate_fn` for the structure-defining entries: atom-wise
    entries are cut along the atom axis at the ``n_atoms`` boundaries, structure-wise ones
    are indexed. Every structure gets its position in the batch as ``properties.idx``, the
    sample index a per-sample transform keys its caches by, so a batch that carries none --
    one read back from a trajectory, say -- still splits into usable samples.

    Neighbor lists are deliberately not split. They are the one thing that cannot be
    recovered by cutting: the pair indices are shifted into batch-global numbering, and any
    caller splitting a batch is about to rebuild them anyway.

    Args:
        inputs: collated input batch.
        keys: entries to split in addition to ``n_atoms``, ``Z``, ``R``, ``cell`` and
            ``pbc``. An entry whose first dimension matches the number of atoms in the
            batch is treated as atom-wise, anything else as structure-wise. The two only
            coincide when every structure holds a single atom, and there the two readings
            cut at the same places anyway.

    Returns:
        list(dict(str, torch.Tensor)): one input dictionary per structure, in batch order.
    """
    n_atoms = inputs[structure.n_atoms]
    n_structures = n_atoms.shape[0]
    n_total_atoms = int(n_atoms.sum())

    offsets = torch.cat(
        [torch.zeros(1, dtype=n_atoms.dtype, device=n_atoms.device), n_atoms.cumsum(0)]
    ).tolist()

    split_keys = list(_STRUCTURE_KEYS) + [
        key for key in (keys or ()) if key not in _STRUCTURE_KEYS
    ]

    samples = []
    for idx in range(n_structures):
        sample = {structure.idx: torch.tensor([idx])}
        for key in split_keys:
            if key not in inputs:
                continue
            value = inputs[key]
            if key == structure.n_atoms:
                sample[key] = value[idx : idx + 1]
            elif value.shape[0] == n_total_atoms:
                sample[key] = value[offsets[idx] : offsets[idx + 1]]
            else:
                sample[key] = value[idx : idx + 1]
        samples.append(sample)

    return samples


class AtomsLoader(DataLoader):
    """Data loader for subclasses of ASEAtomsData"""

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t = _atoms_collate_fn,
        pin_memory: bool = False,
        **kwargs,
    ):
        super(AtomsLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs,
        )

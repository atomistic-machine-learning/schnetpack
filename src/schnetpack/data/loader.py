import torch
from torch.utils.data import DataLoader

from typing import Optional, Sequence
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, T_co

from schnetpack import Structure


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {Structure.idx_i, Structure.idx_j}

    coll_batch = {}
    for key in elem:
        if key not in idx_keys:
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[Structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    coll_batch[Structure.seg_m] = seg_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    return coll_batch


class AtomsLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t = _atoms_collate_fn,
        pin_memory: bool = False,
        **kwargs
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
            **kwargs
        )

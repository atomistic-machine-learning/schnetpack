import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

from schnetpack import Properties
from .stats import StatisticsAccumulator


__all__ = ["AtomsLoader"]


def _collate_aseatoms(examples):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }
    has_atom_mask = Properties.atom_mask in batch.keys()
    has_neighbor_mask = Properties.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] = torch.zeros_like(
            batch[Properties.neighbors]
        ).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] = torch.zeros_like(batch[Properties.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Properties.neighbor_pairs_j in properties:
        batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
            batch[Properties.neighbor_pairs_j]
        ).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Properties.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.neighbor_pairs_mask][s] = nbh_idx_j >= 0

    return batch


class AtomsLoader(DataLoader):
    r"""
    Specialized ``torch.data.DataLoader`` which uses the correct
    collate_fn for AtomsData and provides functionality for calculating mean
    and stddev.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch (default: collate_atons).
        pin_memory (bool, optional): If ``True``, the data loader will copy
            tensors into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete
            batch, if the dataset size is not divisible by the batch size.
            If ``False`` and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for
            collecting a batch from workers. Should always be non-negative.
            (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called
            on each worker subprocess with the worker id (an int in
            ``[0, num_workers - 1]``) as input, after seeding and before data
            loading. (default: None)

    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=_collate_aseatoms,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
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
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

    def get_statistics(
        self, property_names, divide_by_atoms=False, single_atom_ref=None
    ):
        """
        Compute mean and variance of a property. Uses the incremental Welford
        algorithm implemented in StatisticsAccumulator

        Args:
            property_names (str or list): Name of the property for which the
                mean and standard deviation should be computed
            divide_by_atoms (dict or bool): divide mean by number of atoms if True
                (default: False)
            single_atom_ref (dict or bool): reference values for single atoms (default:
                None)

        Returns:
            mean: Mean value
            stddev: Standard deviation

        """
        if type(property_names) is not list:
            property_names = [property_names]
        if type(divide_by_atoms) is not dict:
            divide_by_atoms = {prop: divide_by_atoms for prop in property_names}
        if single_atom_ref is None:
            single_atom_ref = {prop: None for prop in property_names}

        with torch.no_grad():
            statistics = {
                prop: StatisticsAccumulator(batch=True) for prop in property_names
            }
            logger.info("statistics will be calculated...")

            for row in self:
                for prop in property_names:
                    self._update_statistic(
                        divide_by_atoms[prop],
                        single_atom_ref[prop],
                        prop,
                        row,
                        statistics[prop],
                    )

            means = {prop: s.get_mean() for prop, s in statistics.items()}
            stddevs = {prop: s.get_stddev() for prop, s in statistics.items()}

        return means, stddevs

    def _update_statistic(
        self, divide_by_atoms, single_atom_ref, property_name, row, statistics
    ):
        """
        Helper function to update iterative mean / stddev statistics
        """
        property_value = row[property_name]
        if single_atom_ref is not None:
            z = row["_atomic_numbers"]
            p0 = torch.sum(torch.from_numpy(single_atom_ref[z]).float(), dim=1)
            property_value -= p0
        if divide_by_atoms:
            property_value /= torch.sum(row["_atom_mask"], dim=1, keepdim=True)
        statistics.add_sample(property_value)

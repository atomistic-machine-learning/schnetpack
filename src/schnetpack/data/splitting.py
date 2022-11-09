from typing import Optional, List, Dict, Tuple, Union
import math
import torch
import numpy as np

__all__ = ["SplittingStrategy", "RandomSplit", "SubsamplePartitions"]


def absolute_split_sizes(dsize: int, split_sizes: List[int]) -> List[int]:
    """
    Convert partition sizes to absolute values

    Args:
        dsize - Size of dataset.
        split_sizes - Sizes for each split. One can be set to -1 to assign all
            remaining data.
    """
    none_idx = None
    split_sizes = list(split_sizes)
    psum = 0

    for i in range(len(split_sizes)):
        s = split_sizes[i]
        if s is None or s < 0:
            if none_idx is None:
                none_idx = i
            else:
                raise ValueError(
                    f"Only one partition may be undefined (negative or None). "
                    f"Partition sizes: {split_sizes}"
                )
        else:
            if s < 1:
                split_sizes[i] = int(math.floor(s * dsize))

            psum += split_sizes[i]

    if none_idx is not None:
        remaining = dsize - psum
        split_sizes[none_idx] = int(remaining)

    return split_sizes


def random_split(dsize: int, *split_sizes: Union[int, float]) -> List[torch.tensor]:
    """
    Randomly split the dataset

    Args:
        dsize - Size of dataset.
        split_sizes - Sizes for each split. One can be set to -1 to assign all
            remaining data. Values in [0, 1] can be used to give relative partition
            sizes.
    """
    split_sizes = absolute_split_sizes(dsize, split_sizes)
    offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
    indices = torch.randperm(sum(split_sizes)).tolist()
    partition_sizes_idx = [
        indices[offset - length : offset]
        for offset, length in zip(offsets, split_sizes)
    ]
    return partition_sizes_idx


class SplittingStrategy:
    """
    Base class to implement various data splitting methods.
    """

    def __init__(self):
        pass

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        """
        Args:
            dsize - Size of dataset.
            split_sizes - Sizes for each split. One can be set to -1 to assign all
                remaining data. Values in [0, 1] can be used to give relative partition
                sizes.

        Returns:
            list of partitions, where each one is a torch tensor with indices

        """
        raise NotImplementedError


class RandomSplit(SplittingStrategy):
    """
    Splitting strategy that partitions the data randomly into the given sizes
    """

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        dsize = len(dataset)
        partition_sizes_idx = random_split(dsize, *split_sizes)
        return partition_sizes_idx


class SubsamplePartitions(SplittingStrategy):
    """
    Strategy that splits the atoms dataset into predefined partitions as defined in the
    metadata. If the split size is smaller than the predefined partition, a given
    strategy will be used to subsample the partition (default: random).

    An metadata in the atoms dataset might look like this:

    >>> metadata = {
        my_partition_key : {
            "known": [0, 1, 2, 3],
            "test": [5, 6, 7]
        }
     }

    """

    def __init__(
        self,
        split_partition_sources: List[str],
        split_id=0,
        base_splitting: Optional[SplittingStrategy] = None,
        partition_key: str = "splits",
    ):
        """
        Args:
            split_partition_sources: names of partitions in metadata in the order of the
                supplied `split_sizes` in the `split` method. The same source can be
                used for multiple partitions. In that case the given `base_splitting`
                handles distribution the further splitting within each of the sources
                separately.
            split_id: the id of the predefined splitting
            base_splitting: method to subsample each partition
            partition_key: key in the metadata under which teh splitting is stores.
        """
        self.split_partition_sources = split_partition_sources
        self.partition_key = partition_key
        self.split_id = split_id

        self._unique_sources, self._splits_indices = np.unique(
            self.split_partition_sources, return_inverse=True
        )
        self.base_splitting = base_splitting or RandomSplit()

    def split(self, dataset, *split_sizes):
        if len(split_sizes) != len(self.split_partition_sources):
            raise ValueError(
                f"The number of `split_sizes`({len(split_sizes)}) needs to match the "
                + f"number of `partition_sources`({len(self.split_partition_sources)})."
            )

        split_partition_sizes = {src: [] for src in self.split_partition_sources}
        split_partition_idx = {src: [] for src in self.split_partition_sources}
        for i, split_size, src in zip(
            range(len(split_sizes)), split_sizes, self.split_partition_sources
        ):
            split_partition_sizes[src].append(split_size)
            split_partition_idx[src].append(i)

        partitions = dataset.metadata[self.partition_key]

        split_indices = [None] * len(split_sizes)
        for src in self._unique_sources:
            partition = partitions[src][self.split_id]
            print(len(partition))
            partition_split_indices = random_split(
                len(partition), *split_partition_sizes[src]
            )
            for i, split_idx in zip(split_partition_idx[src], partition_split_indices):
                split_indices[i] = np.array(partition)[split_idx].tolist()
        return split_indices

from typing import Optional, List, Dict, Tuple, Union
import math
import torch

__all__ = ["SplittingStrategy", "RandomSplit"]


class SplittingStrategy:
    def __init__(self):
        pass

    def _absolute_split_sizes(self, dataset, partition_sizes):
        """Convert partition sizes to absolute values"""
        none_idx = None
        dsize = len(dataset)
        partition_sizes = list(partition_sizes)
        psum = 0

        for i in range(len(partition_sizes)):
            s = partition_sizes[i]
            if s is None or s < 0:
                if none_idx is None:
                    none_idx = i
                else:
                    raise ValueError(
                        f"Only one partition may be undefined (negative or None). "
                        f"Partition sizes: {partition_sizes}"
                    )
            else:
                if s < 1:
                    partition_sizes[i] = int(math.floor(s * dsize))

                psum += s

        if none_idx is not None:
            remaining = dsize - psum
            partition_sizes[none_idx] = remaining

        return partition_sizes

    def split(self, dataset, *partition_sizes):
        raise NotImplementedError


class RandomSplit(SplittingStrategy):
    """
    Split the data randomly into the given partition sizes
    """

    def __init__(self):
        pass

    def split(self, dataset, *partition_sizes):
        partition_sizes = self._absolute_split_sizes(dataset, partition_sizes)
        offsets = torch.cumsum(torch.tensor(partition_sizes), dim=0)
        indices = torch.randperm(sum(partition_sizes)).tolist()
        partition_sizes_idx = [
            indices[offset - length : offset]
            for offset, length in zip(offsets, partition_sizes)
        ]
        return partition_sizes_idx

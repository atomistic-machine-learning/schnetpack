from typing import Optional, List, Dict, Union
import math
import torch
import numpy as np


__all__ = [
    "SplittingStrategy",
    "RandomSplit",
    "SubsamplePartitions",
    "GroupSplit",
    "ProportionalSplit",
]


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
        split_sizes - Sizes for each split. One value can be set to None to assign all
            remaining data. Values in [0, 1) can be used to give relative partition
            sizes.
    """
    split_sizes = absolute_split_sizes(dsize, split_sizes)
    offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
    indices = torch.randperm(dsize).tolist()
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
            dataset - The dataset that is supposed to be split (an instance of BaseAtomsData).
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


class GroupSplit(SplittingStrategy):
    """
    Strategy that splits the atoms dataset into non-overlapping groups, atoms under the same groups
    (setreoisomers/conformers) will be added to only one of the splits.

    the dictionary of groups is defined in the metadata under the key 'groups_ids' as follows:

    >>> metadata = {
        groups_ids : {
            "smiles_ids": [0, 1, 2, 3],
            "stereo_iso_id": [5, 6, 7],
            ...
        }
     }

    """

    def __init__(
        self,
        splitting_key: str,
        meta_key: str = "groups_ids",
        dataset_ids_key: Optional[str] = None,
    ):
        """
        Args:
            splitting_key: the id's key which will be used for the group splitting.
            meta_key: key in the metadata under which the groups ids and other ids are saved.
            dataset_ids_key: key in the metadata under which the ASE database ids are saved.
        """
        self.splitting_key = splitting_key
        self.meta_key = meta_key
        self.dataset_ids_key = dataset_ids_key

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        md = dataset.metadata

        groups_ids = torch.tensor(md[self.meta_key][self.splitting_key])

        if len(dataset) != len(groups_ids) and dataset.subset_idx is None:
            raise ValueError(
                "The length of the dataset and the length of the groups ids are not equal."
            )

        # if the dataset is a subset of the original dataset, we need to map the groups ids to the subset ids
        if dataset.subset_idx is not None:
            _subset_ids = dataset.subset_idx
        else:
            _subset_ids = torch.arange(len(dataset))

        try:
            groups_ids = groups_ids[_subset_ids]
        except:
            raise ValueError(
                "the subset used of the dataset and the groups ids arrays provided doesn't match."
            )

        # check the split sizes
        unique_groups = torch.unique(groups_ids)
        dsize = len(unique_groups)
        sum_split_sizes = sum([s for s in split_sizes if s is not None and s > 0])

        if sum_split_sizes > dsize:
            raise ValueError(
                f"The sum of the splits sizes '{split_sizes}' should be less than "
                f"the number of available groups '{dsize}'."
            )

        # split the groups
        partitions = random_split(dsize, *split_sizes)
        partitions = [torch.isin(groups_ids, unique_groups[p]) for p in partitions]
        partitions = [(torch.where(p)[0]).tolist() for p in partitions]

        return partitions


class ProportionalSplit(SplittingStrategy):
    """
    Splitting strategy for MergedDataset that preserves a fixed per-dataset
    proportion in every split (train / val / test).

    ##NOTE: ASK STEFAAN
    Sampling is without replacement — no sample appears in more than one split.

    Args:
        proportions: mapping from dataset name to relative weight.
                     Values are normalised to sum=1 internally, so
                     {"md17": 1, "rmd17": 1} == {"md17": 0.5, "rmd17": 0.5}.
        seed: random seed for reproducible sampling.
    """

    def __init__(self, proportions: Dict[str, float], seed: int = 42) -> None:
        super().__init__()
        self.proportions = proportions
        self.seed = seed

    def split(self, dataset, *split_sizes) -> List[List[int]]:
        """
        Args:
            dataset: a MergedDataset instance.
            *split_sizes: sizes for each split (absolute or fractional),
                          forwarded directly from AtomsDataModuleV2.

        Returns:
            List of index lists into dataset.plan, one per split.
        """
        from schnetpack.datasets.merge_db import MergedDataset

        if not isinstance(dataset, MergedDataset):
            raise ValueError(
                "ProportionalSplit only works with MergedDataset instances."
            )

        rng = np.random.default_rng(self.seed)
        dataset_names = list(dataset.datasets.keys())

        missing = [n for n in dataset_names if n not in self.proportions]
        if missing:
            raise ValueError(f"Missing proportions for datasets: {missing}")

        # Normalise proportions
        total = float(sum(self.proportions[n] for n in dataset_names))
        norm = {n: self.proportions[n] / total for n in dataset_names}

        # Resolve fractional sizes to absolute counts
        abs_sizes = absolute_split_sizes(len(dataset), list(split_sizes))

        # Per-dataset counts for each split
        counts_per_split = [
            self._counts_from_proportions(
                size, norm, dataset_names
            )  ## largest-remainder method for safety
            for size in abs_sizes
        ]

        # Build per-name pools: positions in dataset.plan
        plan_indices_by_name: Dict[str, List[int]] = {n: [] for n in dataset_names}
        for pos, (dataset_name, _) in enumerate(dataset.plan):
            plan_indices_by_name[dataset_name].append(pos)

        # Validate we have enough samples per dataset across all splits
        for name in dataset_names:
            needed = sum(c[name] for c in counts_per_split)
            available = len(plan_indices_by_name[name])
            if needed > available:
                raise ValueError(
                    f"Not enough samples in '{name}': "
                    f"need {needed}, have {available}."
                )

        # Sample without replacement then slice into splits
        result: List[List[int]] = [[] for _ in abs_sizes]

        for name in dataset_names:
            pool = np.array(plan_indices_by_name[name])
            total_needed = sum(c[name] for c in counts_per_split)
            chosen = rng.choice(pool, size=total_needed, replace=False)

            offset = 0
            for split_idx, counts in enumerate(counts_per_split):
                n = counts[name]
                result[split_idx].extend(chosen[offset : offset + n].tolist())
                offset += n

        # Shuffle each split so datasets are interleaved, not blocked by source
        for split_indices in result:
            rng.shuffle(split_indices)

        return result

    @staticmethod
    def _counts_from_proportions(
        split_size: int,
        proportions: Dict[str, float],
        names: List[str],
    ) -> Dict[str, int]:
        """Largest-remainder allocation of split_size across datasets."""
        raw = {n: proportions[n] * split_size for n in names}
        base = {n: int(np.floor(raw[n])) for n in names}
        remainder = split_size - sum(base.values())

        if remainder > 0:
            order = sorted(names, key=lambda n: raw[n] - base[n], reverse=True)
            for i in range(remainder):
                base[order[i % len(order)]] += 1

        return base


"""
- MD17: 200,000 samples
- rMD17: 80,000 samples
- Total merged: 300,000
- `num_train=0.8` → 240,000 train samples 120 
- `num_val=0.1` → 30,000 val samples
- `num_test=0.1` → 30,000 test samples
- Proportions: `{"md17": 0.7, "rmd17": 0.3}`

Step 1 — Normalise proportions**

md17:  0.7 / (0.7+0.3) = 0.7
rmd17: 0.3 / (0.7+0.3) = 0.3

Step 2 — Figure out how many samples per dataset per split

For train (240,000 total):
md17:  0.7 x 240,000 = 168,000
rmd17: 0.3 x 240,000 =  72,000

For val (30,000 total):
md17:  0.7 x 30,000 = 21,000
rmd17: 0.3 x 30,000 =  9,000

For test (30,000 total):
md17:  0.7 x 30,000 = 21,000
rmd17: 0.3 x 30,000 =  9,000

Step 3 — Check availability
md17  needs: 168,000 + 21,000 + 21,000 = 210,000 — have 200,000  → raises error
rmd17 needs:  72,000 +  9,000 +  9,000 =  90,000 — have 100,000 

Step 4 — Build per-name index pools
plan_indices_by_name = {
    "md17":  [0, 1, 2, ..., 199999],   # positions in the plan
    "rmd17": [200000, 200001, ..., 299999],
}

Step 5 — Sample without replacement
md17 chosen = rng.choice(200000 indices, size=210000)  # error, not enough
rmd17 chosen = rng.choice(100000 indices, size=90000, replace=False)
  → first 72000 go to train
  → next   9000 go to val
  → last   9000 go to test

Step 6 — Shuffle each split
"""

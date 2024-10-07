from typing import Optional, List, Dict, Tuple, Union
import math
import torch
import numpy as np

__all__ = ["SplittingStrategy", "RandomSplit", "SubsamplePartitions","AtomTypeSplit"]


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


class AtomTypeSplit(SplittingStrategy):

    """
    Strategy that filters out a specific atom type or multiple atom types from the dataset.
    And then performs the splitting on the filtered dataset.
    Requires the metadata to contain the atom type information (list of lists converted to ndarray).
    """

    def __init__(self,atomtypes: List[int], percentage_to_keep: float = 0.0):
        """
        Args:
            atomtypes: list of atom types to be filtered out
            percentage_to_keep: percentage of the to be filtered out atomtypes to keep.
                For now the percentage is applied to all atomtypes.
        """
        self.atomtypes = atomtypes
        self.percentage_to_keep = percentage_to_keep
    
    def calc_percentage_to_keep(self, dataset):
        pass

    def split(self, dataset, *split_sizes):

        atom_type_count = np.array(dataset.conn.metadata["atom_type_count"])
        mask = (atom_type_count[:, self.atomtypes] == 0).all(axis=1)
        indices = np.where(mask)[0].tolist() #+ 1

        if self.percentage_to_keep > 0:
            # how many occurences of atom type in dataset, and how many to keep
            # keeping means how many absolute occurences to keep, not relative, e.g 5% of all carbon occurences of the complete dataset are kept
            if len(self.atomtypes) == 1:
                total_atom_type_count = np.round(atom_type_count[:,self.atomtypes].sum() * self.percentage_to_keep)

                # init variables to track the running sum and indices
                current_sum = 0
                # random indices to iterate through the array to find occurences to keep
                random_iter_indices = np.random.permutation(len(atom_type_count)).tolist()
                random_iter_indices = [n for n in random_iter_indices if n not in indices]
                selected_indices = []

                # iter through, to find the subset that sums up to number of occurences to keep
                for i in random_iter_indices:
                    value = atom_type_count[i,self.atomtypes]
                    if current_sum + value <= total_atom_type_count.item():
                        current_sum += value
                        selected_indices.append(i)
                    if current_sum >= total_atom_type_count.item():
                        break
            # logic should mainly apply only to TM metals because they are only sparsly represented in the dataset
            else:
                # select all indices that are not in the mask
                selected_indices = np.where(~mask)[0]
                # num of molecules to keep
                num_to_keep = int(len(selected_indices) * self.percentage_to_keep)
                # shuffle to circumvent to sample highly represented TM
                permuted_indices = selected_indices[torch.randperm(len(selected_indices))]
                selected_indices = permuted_indices[:num_to_keep].tolist()

            #mutually_exclusive = set(indices).isdisjoint(set(selected_indices))
            indices.extend(selected_indices)
            #indices = np.array(indices)

        partition_sizes_idx = self.random_split(np.array(indices), *split_sizes)
        return partition_sizes_idx

    def random_split(self,indices, *split_sizes: Union[int, float]) -> List[torch.tensor]:
        """
        Randomly split the dataset

        Args:
            dsize - Size of dataset.
            split_sizes - Sizes for each split. One can be set to -1 to assign all
                remaining data. Values in [0, 1] can be used to give relative partition
                sizes.
        """
        dsize = len(indices)
        split_sizes = absolute_split_sizes(dsize, split_sizes)
        offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
        indices = indices[torch.randperm(len(indices)).tolist()].tolist()
        partition_sizes_idx = [
            indices[offset - length : offset]
            for offset, length in zip(offsets, split_sizes)
        ]
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

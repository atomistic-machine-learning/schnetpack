from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized, Callable
import torch
import random
from schnetpack import properties
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
import matplotlib.pyplot as plt
from schnetpack.data import BaseAtomsData


__all__ = [
    "WeightedSampler",
    "TrajectorySampler",
    "WeightedTrajectorySampler",
    "TrajectoryFirstDatapointSampler",
    "StratumWeights",
    "TipHeightWeights",
]


class StratumWeights:
    """
    Dummy Calculator for stratum weights that returns an uniform weights distribution for all samples (weight of each
    sample is close to 1).
    """
    def __init__(self, dataset, num_bins):
        self.dataset = dataset
        self.num_bins = num_bins

    def calculate(self):
        feature_values = self._partition_criterion()
        min_value = min(feature_values)
        max_value = max(feature_values)

        bins_array = np.linspace(min_value, max_value, num=self.num_bins + 1)[1:]
        bins_array[-1] += 0.1
        bin_indices = np.digitize(feature_values, bins_array)
        bin_counts = np.bincount(bin_indices, minlength=self.num_bins)

        min_counts = min(bin_counts[bin_counts != 0])
        bin_weights = np.where(bin_counts == 0, 0, min_counts / bin_counts)

        weights = np.zeros(len(self.dataset))
        for i, idx in enumerate(bin_indices):
            weights[i] = bin_weights[idx]

        return weights

    def _partition_criterion(self) -> list:
        values = []
        for spl_idx in range(len(self.dataset)):
            values.append(spl_idx)
        return values


class TipHeightWeights(StratumWeights):
    """
    ...
    """
    def __init__(self, dataset, num_bins):
        super().__init__(dataset=dataset, num_bins=num_bins)

    def _partition_criterion(self) -> list:

        at_idx = 114

        values = []
        for spl_idx in range(len(self.dataset)):
            data = self.dataset[spl_idx]
            values.append(data[properties.R][at_idx, 2])
        return values


def stratified_weights(dataset, partition_criterion, num_bins=10):
    feature_values = partition_criterion(dataset)
    min_value = min(feature_values)
    max_value = max(feature_values)

    bins_array = np.linspace(min_value, max_value, num=num_bins + 1)[1:]
    bins_array[-1] += 0.1
    bin_indices = np.digitize(feature_values, bins_array)
    bin_counts = np.bincount(bin_indices, minlength=num_bins)

    min_counts = min(bin_counts[bin_counts != 0])
    bin_weights = np.where(bin_counts == 0, 0, min_counts / bin_counts)

    weights = np.zeros(len(dataset))
    for i, idx in enumerate(bin_indices):
        weights[i] = bin_weights[idx]

    return weights


def tip_heights(dataset: BaseAtomsData) -> list:
    at_idx = 114

    values = []
    for spl_idx in range(len(dataset)):
        data = dataset[spl_idx]
        values.append(data[properties.R][at_idx, 2])
    return values


class StratifiedSampler(Sampler):
    """
    Note: make sure that num_bins is chosen sufficiently small to avoid too many empty bins.
    """
    def __init__(
            self,
            data_source: BaseAtomsData,
            partition_criterion: Callable[[BaseAtomsData], List],
            num_samples: int,
            num_bins: int = 10,
            replacement: bool = False,
            verbose: bool = True,
    ) -> None:
        self.data_source = data_source
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.replacement = replacement
        self.verbose = verbose

        self.weights = self.calculate_weights(partition_criterion)
        self._sample_counter = [0 for _ in range(len(data_source))]

    def calculate_weights(self, partition_criterion):

        feature_values = partition_criterion(self.data_source)
        min_value = min(feature_values)
        max_value = max(feature_values)

        bins_array = np.linspace(min_value, max_value, num=self.num_bins + 1)[1:]
        bins_array[-1] += 0.1
        bin_indices = np.digitize(feature_values, bins_array)
        bin_counts = np.bincount(bin_indices, minlength=self.num_bins)

        min_counts = min(bin_counts[bin_counts != 0])
        bin_weights = np.where(bin_counts == 0, 0, min_counts / bin_counts)

        weights = np.zeros(len(self.data_source))
        for i, idx in enumerate(bin_indices):
            weights[i] = bin_weights[idx]

        return weights

    def __iter__(self) -> Iterator[int]:

        print("iter=======================")

        # compute weighted selection of ids
        ids = torch.multinomial(torch.tensor(self.weights), self.num_samples, self.replacement)

        if self.verbose:
            # print selection
            selection = [0 for _ in range(len(self._sample_counter))]

            # update id counts
            for idx in ids:
                self._sample_counter[idx] += 1
                selection[idx] += 1

            print("selection:")
            print(selection)
            print("total used:")
            print(self._sample_counter)

        yield from iter(ids.tolist())

    def __len__(self):
        return self.num_samples


class WeightedSampler(Sampler):

    def __init__(self, data_source, num_samples, replacement: bool = True) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self._sample_counter = [0 for _ in range(len(self.data_source))]

    def __iter__(self) -> Iterator[int]:
        print("iter=======================")
        # add ids for new datapoints
        self._update_ids()

        # compute weighted selection of ids
        max_counts = max(self._sample_counter)
        weights = [1 - 0.5*w/(max_counts+1) for w in self._sample_counter]
        ids = torch.multinomial(torch.tensor(weights), self._num_samples, self.replacement)

        # print selection
        selection = [0 for _ in range(len(self._sample_counter))]

        # update id counts
        for idx in ids:
            self._sample_counter[idx] += 1
            selection[idx] += 1

        print("selection:")
        print(selection)
        print("total used:")
        print(self._sample_counter)

        yield from iter(ids.tolist())

    def _update_ids(self):
        n_ids = len(self.data_source)
        n_old_samples = len(self._sample_counter)
        self._sample_counter += [0 for _ in range(n_ids-n_old_samples)]


class TrajectorySampler(Sampler):

    def __init__(self, data_source, num_samples, update_mapping=False, shuffle=True) -> None:
        self.data_source = data_source
        self.num_samples = num_samples
        self.update_mapping = update_mapping
        self.mapping = None
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        # update mapping if required
        if self.mapping is None or self.update_mapping:
            self.mapping = self._get_traj_idx_to_db_idx()

        # sample from trajectories
        num_samples_per_traj = max(1, round(self.num_samples/len(self.mapping.keys())))
        ids = []
        for structure_idx, dataset_ids in self.mapping.items():
            if len(dataset_ids) < self.num_samples:
                ids.extend(random.choices(dataset_ids, k=num_samples_per_traj))
            else:
                ids.extend(random.sample(dataset_ids, k=num_samples_per_traj))

        print(
            f"sampled from {len(self.mapping.keys())} trajectories with "
            f"{num_samples_per_traj} datapoints per trajectory."
        )

        if self.shuffle:
            random.shuffle(ids)

        yield from iter(ids)

    def _get_traj_idx_to_db_idx(self):
        """
        Get a mapping from trajectory ids to dataset ids.

        """
        mapping = dict()
        for dataset_idx, properties in enumerate(
                self.data_source.iter_properties(load_properties=["structure_idx"])
        ):
            structure_idx = int(properties["structure_idx"].item())
            if structure_idx not in mapping.keys():
                mapping[structure_idx] = []
            mapping[structure_idx].append(dataset_idx)

        return mapping


class WeightedTrajectorySampler(TrajectorySampler):

    def __init__(self, data_source, num_samples, update_mapping=False, shuffle=True) -> None:
        super(WeightedTrajectorySampler, self).__init__(
            data_source=data_source,
            num_samples=num_samples,
            update_mapping=update_mapping,
            shuffle=shuffle,
        )

    def __iter__(self) -> Iterator[int]:
        # update mapping if required
        if self.mapping is None or self.update_mapping:
            self.mapping = {k: sorted(v) for k, v in self._get_traj_idx_to_db_idx().items()}

        # sample from trajectories
        num_samples_per_traj = max(1, round(self.num_samples/len(self.mapping.keys())))
        ids = []
        for structure_idx, dataset_ids in self.mapping.items():
            weights = 1 / torch.linspace(1, len(dataset_ids), len(dataset_ids))

            ids.extend(
                [dataset_ids[idx] for idx in torch.multinomial(torch.tensor(weights), num_samples_per_traj, replacement=True,)]
            )

        print(
            f"sampled from {len(self.mapping.keys())} trajectories with "
            f"{num_samples_per_traj} datapoints per trajectory."
        )

        if self.shuffle:
            random.shuffle(ids)

        yield from iter(ids)


class TrajectoryFirstDatapointSampler(TrajectorySampler):

    def __init__(self, data_source, update_mapping=False, shuffle=False):
        super(TrajectoryFirstDatapointSampler, self).__init__(
            data_source=data_source,
            num_samples=None,
            update_mapping=update_mapping,
            shuffle=shuffle,
        )

    def __iter__(self) -> Iterator[int]:
        # update mapping if required
        if self.mapping is None or self.update_mapping:
            self.mapping = {k: sorted(v) for k, v in self._get_traj_idx_to_db_idx().items()}

        # sample from trajectories
        ids = [dataset_ids[0] for dataset_ids in self.mapping.values()]

        print(
            f"sampled first datapoint from {len(self.mapping.keys())} trajectories"
        )

        if self.shuffle:
            random.shuffle(ids)

        yield from iter(ids)

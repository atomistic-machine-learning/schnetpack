from typing import Iterator, List, Callable

import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler

from schnetpack import properties
from schnetpack.data import BaseAtomsData


__all__ = [
    "StratifiedSampler",
    "NumberOfAtomsCriterion",
    "PropertyCriterion",
]


class NumberOfAtomsCriterion:
    """
    A callable class that returns the number of atoms for each sample in the dataset.
    """

    def __call__(self, dataset):
        n_atoms = []
        for spl_idx in range(len(dataset)):
            sample = dataset[spl_idx]
            n_atoms.append(sample[properties.n_atoms].item())
        return n_atoms


class PropertyCriterion:
    """
    A callable class that returns the specified property for each sample in the dataset.
    Property must be a scalar value.
    """

    def __init__(self, property_key: str = properties.energy):
        self.property_key = property_key

    def __call__(self, dataset):
        property_values = []
        for spl_idx in range(len(dataset)):
            sample = dataset[spl_idx]
            property_values.append(sample[self.property_key].item())
        return property_values


class StratifiedSampler(WeightedRandomSampler):
    """
    A custom sampler that performs stratified sampling based on a partition criterion.

    Note: Make sure that num_bins is chosen sufficiently small to avoid too many empty bins.
    """

    def __init__(
        self,
        data_source: BaseAtomsData,
        partition_criterion: Callable[[BaseAtomsData], List],
        num_samples: int,
        num_bins: int = 10,
        replacement: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            data_source: The data source to be sampled from.
            partition_criterion: A callable function that takes a data source
                and returns a list of values used for partitioning.
            num_samples: The total number of samples to be drawn from the data source.
            num_bins: The number of bins to divide the partitioned values into. Defaults to 10.
            replacement: Whether to sample with replacement or without replacement. Defaults to True.
            verbose: Whether to print verbose output during sampling. Defaults to True.
        """
        self.data_source = data_source
        self.num_bins = num_bins
        self.verbose = verbose

        weights = self.calculate_weights(partition_criterion)
        super().__init__(
            weights=weights, num_samples=num_samples, replacement=replacement
        )

    def calculate_weights(self, partition_criterion):
        """
        Calculates the weights for each sample based on the partition criterion.
        """
        feature_values = partition_criterion(self.data_source)

        bin_counts, bin_edges = np.histogram(feature_values, bins=self.num_bins)
        bin_edges = bin_edges[1:]
        bin_edges[-1] += 0.1
        bin_indices = np.digitize(feature_values, bin_edges)

        min_counts = min(bin_counts[bin_counts != 0])
        bin_weights = np.where(bin_counts == 0, 0, min_counts / bin_counts)
        weights = bin_weights[bin_indices]

        return weights

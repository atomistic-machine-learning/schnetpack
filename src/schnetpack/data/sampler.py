from typing import Iterator, List, Callable
import torch
import random
from schnetpack import properties
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
from schnetpack.data import BaseAtomsData


__all__ = [
    "StratifiedSampler",
    "uniform_weights",
]


def uniform_weights(self) -> list:
    """
    Dummy Calculator for stratum weights that returns a uniform weights distribution for all samples (weight of each
    sample is close to 1).
    """
    values = []
    for spl_idx in range(len(self.dataset)):
        values.append(spl_idx)
    return values


class StratifiedSampler(WeightedRandomSampler):
    """
    Note: make sure that num_bins is chosen sufficiently small to avoid too many empty bins.
    All str arguments must correspond to python classes
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
        self.data_source = data_source
        self.num_bins = num_bins
        self.verbose = verbose

        weights = self.calculate_weights(partition_criterion)
        super().__init__(weights=weights, num_samples=num_samples, replacement=replacement)

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

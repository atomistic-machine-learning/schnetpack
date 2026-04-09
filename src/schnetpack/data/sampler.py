from typing import Iterator, List, Callable, Dict

import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
import torch


from schnetpack import properties
from schnetpack.data import ASEAtomsData


__all__ = [
    "StratifiedSampler",
    "NumberOfAtomsCriterion",
    "PropertyCriterion",
    "DatasetBalancedSampler",
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
        data_source: ASEAtomsData,
        partition_criterion: Callable[[ASEAtomsData], List],
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


class DatasetBalancedSampler(WeightedRandomSampler):
    """
    Weighted sampler that balances sampling across component datasets in a
    MergedDataset according to target proportions.

    Note: replacement=True (default) is required when upsampling the smaller dataset.
    """

    def __init__(
        self,
        data_source,
        num_samples: int,
        proportions: Dict[str, float],
        replacement: bool = True,
    ) -> None:
        """
        Args:
            data_source: a MergedDataset instance (or any dataset with a
                         ``plan`` attribute of List[Tuple[str, int]]).
            num_samples: total number of samples to draw per epoch.
                         AtomsDataModuleV2._setup_sampler passes len(dataset).
            proportions: target proportion per dataset name. Normalised
                         internally so {"md17": 1, "rmd17": 1} == 50/50.
            replacement: sample with replacement. Must be True when any
                         dataset is being upsampled. Default: True.
        """
        if not hasattr(data_source, "plan"):
            raise ValueError(
                "DatasetBalancedSampler requires a MergedDataset with a "
                "'plan' attribute."
            )

        self.data_source = data_source
        self.proportions = proportions

        weights = self._calculate_weights(data_source, proportions)

        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
        )

    @staticmethod
    def _calculate_weights(
        dataset,
        proportions: Dict[str, float],
    ) -> torch.Tensor:
        """
        Assign a sampling weight to each sample in dataset.plan.

        Weight formula:
            weight[i] = target_proportion[dataset_name] / count[dataset_name]

        Example:
            md17  count=80,  target=0.5 → each md17  sample weight = 0.5/80  = 0.00625
            rmd17 count=800, target=0.5 → each rmd17 sample weight = 0.5/800 = 0.000625
            → md17 samples are drawn 10x more often, achieving 50/50 balance.
        """
        # Normalise proportions
        total = float(sum(proportions.values()))
        norm = {k: v / total for k, v in proportions.items()}

        # Count how many samples from each dataset are in the plan
        counts: Dict[str, int] = {}
        for dataset_name, _ in dataset.plan:
            counts[dataset_name] = counts.get(dataset_name, 0) + 1

        # Validate all dataset names in plan are covered by proportions
        missing = [n for n in counts if n not in norm]
        if missing:
            raise ValueError(
                f"DatasetBalancedSampler: no proportion specified for "
                f"datasets: {missing}. Add them to proportions dict."
            )

        # Assign weight to each plan entry
        weights = torch.zeros(len(dataset.plan), dtype=torch.double)
        for i, (dataset_name, _) in enumerate(dataset.plan):
            weights[i] = norm[dataset_name] / counts[dataset_name]

        return weights

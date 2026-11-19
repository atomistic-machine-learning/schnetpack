from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from schnetpack import properties

from .atoms import ASEAtomsData

__all__ = [
    "ChunkedRandomSampler",
    "StratifiedSampler",
    "NumberOfAtomsCriterion",
    "PropertyCriterion",
    "BudgetBatches",
    "OrderedBatches",
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


class ChunkedRandomSampler(Sampler[int]):
    """
    Random sampler that splits each pass over the dataset into shorter epochs.

    One random permutation of the dataset is consumed in chunks of `num_samples`
    indices, one chunk per epoch, and reshuffled once it runs out. Unlike
    `RandomSampler(num_samples=...)`, every sample is visited exactly once per full
    pass, so per-epoch machinery (LR schedulers, checkpointing, early stopping) can
    run more often than once per pass over large datasets.

    Note: epoch-counted settings (e.g. `patience`, `max_epochs`) count short epochs,
    so scale them accordingly. The permutation is not checkpointed; resuming
    training starts a fresh one.
    """

    def __init__(self, data_source: ASEAtomsData, num_samples: int) -> None:
        """
        Args:
            data_source: The data source to be sampled from.
            num_samples: The number of samples drawn per epoch.
        """
        self.data_source = data_source
        self.num_samples = num_samples
        self._permutation = torch.empty(0, dtype=torch.long)
        self._cursor = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            if self._cursor == len(self._permutation):
                self._permutation = torch.randperm(len(self.data_source))
                self._cursor = 0
            yield int(self._permutation[self._cursor])
            self._cursor += 1


class StratifiedSampler(WeightedRandomSampler):
    """
    A custom sampler that performs stratified sampling based on a partition criterion.

    Note: Make sure that num_bins is chosen sufficiently small to avoid too many empty bins.
    """

    def __init__(
        self,
        data_source: ASEAtomsData,
        partition_criterion: Callable[[ASEAtomsData], list],
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


class BudgetBatches:
    """
    Batches packed to an atom and a pair budget instead of a fixed number of
    structures: the structure indices of each training batch, from the atom and pair
    counts alone. Structures are taken from a shuffled order while the batch's atoms
    and pairs fit the budgets, so a batch of large structures holds fewer of them, and
    a batch padded to the budgets holds little padding. The budgets are what
    ``batch_size`` structures hold on average; a batch holds at most
    ``max_structures``, 1.3 times ``batch_size``.

    Every pass yields the same number of batches, as a fixed epoch length needs: the
    batches of one packing, without its last one if ``drop_last``. A pass that packs
    fewer batches is topped up from a second order.
    """

    def __init__(
        self,
        n_atoms,
        n_pairs,
        indices,
        batch_size: int,
        seed: int = 0,
        grains=(32, 256),
        drop_last: bool = True,
    ):
        """
        Args:
            n_atoms: atoms of every structure of the dataset
            n_pairs: neighbor pairs of every structure of the dataset
            indices: the structures to draw from (e.g. the training split)
            batch_size: structures per batch on average
            seed: seed of the orders
            grains: the budgets are rounded up to these (atoms, pairs)
            drop_last: drop the last batch of each packing (usually not full), also
                when it is full. False keeps it: every structure once per pass
        """
        self.indices = np.asarray(indices)
        self.rng = np.random.default_rng(seed)
        self.atoms, self.pairs = np.asarray(n_atoms), np.asarray(n_pairs)
        budgets = []
        for per, grain in ((self.atoms, grains[0]), (self.pairs, grains[1])):
            x = per[self.indices]
            size = max(batch_size * float(x.mean()), float(x.max()), 1.0)
            budgets.append(int(-(-size // grain) * grain))
        self.atom_budget, self.pair_budget = budgets
        self.max_structures = int(np.ceil(1.3 * batch_size))
        self.drop_last = drop_last
        self.n_batches = len(self._pack(self.rng.permutation(self.indices)))

    def _pack(self, order) -> list[list[int]]:
        batches, current, atoms, pairs = [], [], 0, 0
        for s, a, p in zip(
            order.tolist(), self.atoms[order].tolist(), self.pairs[order].tolist()
        ):
            full = len(current) == self.max_structures
            if current and (
                full or atoms + a > self.atom_budget or pairs + p > self.pair_budget
            ):
                batches.append(current)
                current, atoms, pairs = [], 0, 0
            current.append(s)
            atoms, pairs = atoms + a, pairs + p
        batches.append(current)  # the last one, usually not full
        # fewer structures than one batch holds: that one batch
        return batches[:-1] if self.drop_last and len(batches) > 1 else batches

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        batches = []
        while len(batches) < self.n_batches:
            batches += self._pack(self.rng.permutation(self.indices))
        for b in batches[: self.n_batches]:
            yield np.asarray(b, dtype=np.int64)


class OrderedBatches:
    """Batches of all structures in their order, the last one possibly smaller
    (validation and testing): the structure indices of each batch."""

    def __init__(self, indices, batch_size: int):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return -(-len(self.indices) // self.batch_size)

    def __iter__(self):
        for s in range(0, len(self.indices), self.batch_size):
            yield self.indices[s : s + self.batch_size]

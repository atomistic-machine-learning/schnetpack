"""
Virtual merged dataset that combines multiple ASEAtomsData-backed datasets
on the fly — no merged DB is written to disk.

Key design decisions
--------------------
- Each component dataset is fully independent: its own properties, units,
  distance unit, atomrefs, and transforms. No compatibility checks.
- Transforms live on the component datasets, not on MergedDataset.
  MergedDataset.__getitem__ just fetches the already-transformed sample.
- subset_sizes controls how many samples are randomly drawn from each
  component at construction time (without replacement).
- dataset_id and source_index are injected into every sample so downstream
  models and samplers know which database a sample came from.
- Splitting and weighted sampling are handled externally by
  ProportionalSplit and DatasetBalancedSampler.

"""

import copy
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from schnetpack.data.atoms import ASEAtomsData, AtomsDataError

__all__ = ["MergedDataset"]


class MergedDataset(Dataset):
    """
    Virtual merged dataset.

    Per-sample fields injected into every sample dict:
        dataset_id   (int64, shape [1]) — stable id per component (insertion order)
        source_index (int64, shape [1]) — original index in the component dataset
    """

    def __init__(
        self,
        datasets: Dict[str, ASEAtomsData],
        subset_sizes: Optional[Dict[str, int]] = None,
        seed: int = 42,
        add_source_index: bool = True,
    ) -> None:
        """
        Args:
            datasets: component datasets keyed by an arbitrary name.
                      Insertion order determines dataset_id (0, 1, 2, …).
                      Each dataset handles its own transforms independently.
            subset_sizes: number of samples to randomly draw from each dataset
                          without replacement. If None, all samples are used.
                          Keys must match datasets.
            seed: random seed for reproducible subset sampling.
            add_source_index: inject ``source_index`` into each sample.
        """
        self.transforms = []
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        if not datasets:
            raise AtomsDataError("datasets must not be empty.")

        if subset_sizes is not None:
            missing = [n for n in datasets if n not in subset_sizes]
            if missing:
                raise AtomsDataError(
                    f"subset_sizes missing keys for datasets: {missing}"
                )
            for name, size in subset_sizes.items():
                available = len(datasets[name])
                if size > available:
                    warnings.warn(
                        f"subset_sizes['{name}']={size} exceeds available samples "
                        f"({available}). Using all {available} samples instead. "
                        f"DatasetBalancedSampler will compensate via weighting.",
                        UserWarning,
                        stacklevel=2,
                    )
                    subset_sizes[name] = available

        self.datasets = datasets
        self.subset_sizes = subset_sizes
        self.seed = seed
        self.add_source_index = add_source_index

        # Stable integer id per dataset name, in insertion order
        self._dataset_ids: Dict[str, int] = {name: i for i, name in enumerate(datasets)}

        # Build flat plan: (dataset_name, index_in_component)
        self.plan: List[Tuple[str, int]] = self._build_plan(seed)

        # Set by subset() — required by AtomsDataModuleV2
        self.split: Optional[str] = None

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def _build_plan(self, seed: int) -> List[Tuple[str, int]]:
        """
        Build the flat plan by randomly sampling subset_sizes indices
        from each component dataset (without replacement).
        If subset_sizes is None, use all indices in order.
        """
        rng = np.random.default_rng(seed)
        plan: List[Tuple[str, int]] = []

        for name, ds in self.datasets.items():
            n = len(ds)
            if self.subset_sizes is not None:
                size = self.subset_sizes[name]
                chosen = rng.choice(n, size=size, replace=False).tolist()
            else:
                chosen = list(range(n))

            plan.extend((name, int(idx)) for idx in chosen)

        return plan

    # ------------------------------------------------------------------
    # subset() — called by AtomsDataModuleV2.setup() to build
    # train / val / test views
    # ------------------------------------------------------------------

    def subset(
        self,
        subset_idx: List[int],
        split: Optional[str] = None,
    ) -> "MergedDataset":
        """
        Return a shallow copy restricted to the given indices into self.plan.
        """
        ds = copy.copy(self)
        ds.plan = [self.plan[i] for i in subset_idx]
        ds.split = split
        return ds

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        dataset_name, index = self.plan[i]
        component_ds = self.datasets[dataset_name]

        # Each component applies its own transforms internally.
        # No bypassing needed — transforms are fully independent per dataset.
        sample = component_ds[index]

        # Inject provenance
        sample["dataset_id"] = torch.tensor(
            [self._dataset_ids[dataset_name]], dtype=torch.long
        )
        if self.add_source_index:
            sample["source_index"] = torch.tensor([index], dtype=torch.long)

        return sample

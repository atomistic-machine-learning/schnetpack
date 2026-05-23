"""
Virtual merged dataset that combines multiple ASEAtomsData-backed datasets
on the fly — no merged DB is written to disk.

Key design decisions
--------------------
- Each component dataset is fully independent: its own properties, units,
  distance unit, atomrefs, and transforms.
- Transforms live on the component datasets, not on MergedDataset.
- subset_sizes controls how many samples are randomly drawn from each
  component at construction time (without replacement).
- If subset_sizes[name] > len(dataset[name]), a warning is issued and
  all available samples are used. DatasetBalancedSampler compensates.
- dataset_id and source_index are injected into every sample.
- initialize_transforms() is overridden to build per-component providers
  via MergedStatsAtomrefProvider and initialize each component's transforms
  with its own stats.
"""

import copy
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from schnetpack.data.atoms import ASEAtomsData, AtomsDataError
from schnetpack.transform.atomistic import ConditionalRemoveOffsets

__all__ = ["MergedDataset"]


class MergedDataset(Dataset):
    """
    Virtual merged dataset.
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
                      Each dataset carries its own transforms independently.
            subset_sizes: number of samples to randomly draw from each dataset
                          without replacement. If None, all samples are used.
                          If size > available, warns and uses all available.
            seed: random seed for reproducible subset sampling.
            add_source_index: inject ``source_index`` into each sample.
        """
        if not datasets:
            raise AtomsDataError("datasets must not be empty.")

        # Validate and cap subset_sizes
        if subset_sizes is not None:
            missing = [n for n in datasets if n not in subset_sizes]
            if missing:
                raise AtomsDataError(
                    f"subset_sizes missing keys for datasets: {missing}"
                )
            for name, size in list(subset_sizes.items()):
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

        # Required by AtomsDataModule.setup() — always empty on MergedDataset
        # since transforms live on component datasets
        self.transforms = []
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        # Set by subset() — used by component datasets' split-aware dispatch
        self.split: Optional[str] = None

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def _build_plan(self, seed: int) -> List[Tuple[str, int]]:
        """
        Randomly sample subset_sizes indices from each component dataset
        without replacement and concatenate into a flat plan.
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
    # subset() — called by AtomsDataModule.setup()
    # ------------------------------------------------------------------

    def subset(
        self,
        subset_idx: List[int],
        split: Optional[str] = None,
    ) -> "MergedDataset":
        """
        Return a shallow copy restricted to the given indices into self.plan.
        Sets split on each component dataset so their split-aware transform
        dispatch works correctly.
        """
        ds = copy.copy(self)
        ds.plan = [self.plan[i] for i in subset_idx]
        ds.split = split

        # propagate split to component datasets so their get_split_transforms()
        # dispatches correctly (train_transforms vs val_transforms vs transforms)
        for component_ds in ds.datasets.values():
            component_ds.split = split

        return ds

    # ------------------------------------------------------------------
    # initialize_transforms
    # ------------------------------------------------------------------

    def initialize_transforms(self, provider=None) -> None:
        """
        Initialize transforms on each component dataset.
        """
        from schnetpack.data.provider import MergedStatsAtomrefProvider

        # Track already-initialized transform instances to avoid double-init
        # when both datasets share the same transform object (e.g. ${data.transforms})
        initialized = set()

        for name, ds in self.datasets.items():
            for transform in getattr(ds, "transforms", []):
                if not hasattr(transform, "initialize"):
                    continue

                # Skip if this exact instance was already initialized
                if id(transform) in initialized:
                    continue
                initialized.add(id(transform))

                if isinstance(transform, ConditionalRemoveOffsets):
                    # Needs full MergedStatsAtomrefProvider — not per-component
                    try:
                        transform.initialize(provider)
                    except TypeError as e:
                        raise TypeError(
                            f"ConditionalRemoveOffsets.initialize() failed for "
                            f"dataset '{name}': {e}"
                        ) from e

                elif isinstance(provider, MergedStatsAtomrefProvider):
                    # All other transforms get the per-component provider
                    component_provider = provider.providers.get(name)
                    try:
                        transform.initialize(component_provider)
                    except TypeError:
                        transform.initialize()

                else:
                    try:
                        transform.initialize(provider)
                    except TypeError:
                        transform.initialize()

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        dataset_name, index = self.plan[i]
        component_ds = self.datasets[dataset_name]
        dataset_id = self._dataset_ids[dataset_name]

        # Load raw properties WITHOUT applying transforms yet
        actual_idx = (
            component_ds.subset_idx[index]
            if component_ds.subset_idx is not None
            else index
        )
        props = component_ds._get_properties(
            component_ds.conn,
            actual_idx,
            component_ds.load_properties,
            component_ds.load_structure,
        )

        # Inject provenance BEFORE transforms so ConditionalRemoveOffsets
        # can find _source_index in forward()
        props["dataset_id"] = torch.tensor([dataset_id], dtype=torch.long)
        if self.add_source_index:
            props["source_index"] = torch.tensor([dataset_id], dtype=torch.long)
            # _source_index = dataset_id (0=md17, 1=rmd17), NOT the sample index
            # This is what ConditionalRemoveOffsets and ConditionalAddOffsets route on

        # Now apply transforms with _source_index already present
        props = component_ds._apply_transforms(props)

        return props

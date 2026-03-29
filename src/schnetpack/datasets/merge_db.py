"""
MergedDataset deliberately does not subclass ASEAtomsData because there is no
backing ASE DB file.  Instead it mirrors the complete public interface that
AtomsDataModuleV2 (and StatsAtomrefProvider) depend on:

    __len__, __getitem__
    transforms / train_transforms / val_transforms / test_transforms
    subset(idx, split)          — returns a new MergedDataset view
    atomrefs                    — None or Dict[str, torch.Tensor]
    available_properties        — union of component properties
    load_properties             — settable filter (validated against available)
    units                       — property → unit string dict
    distance_unit               — single string (must agree across components)
    split                       — "train" | "val" | "test" | None

Usage
-----
    qm9  = QM9(...)
    rmd  = RMD17(...)

    merged = MergedDataset.from_datasets(
        datasets    = {"qm9": qm9, "rmd17": rmd},
        proportions = {"qm9": 0.7, "rmd17": 0.3},
        total_size  = 10_000,
        num_train   = 0.8,
        num_val     = 0.1,
        seed        = 42,
    )

    dm = AtomsDataModuleV2(
        dataset   = merged,
        batch_size = 32,
        num_train  = merged.num_train,
        num_val    = merged.num_val,
        num_test   = merged.num_test,
        split_file = None,           # MergedDataset owns its own plan
    )
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from schnetpack.data.atoms import ASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["MergedDataset"]

SplitSize = Union[int, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_split_sizes(
        total_size: int,
        num_train: SplitSize,
        num_val: SplitSize,
        num_test: Optional[SplitSize],
    ) -> Tuple[int, int, int]:
    """
    Convert (possibly fractional) split specs to absolute counts.
    """

    def _to_abs(x: SplitSize) -> int:
        if isinstance(x, float) and x <= 1.0:
            return int(round(x * total_size))
        return int(x)

    if num_train is None or num_val is None:
        raise ValueError("num_train and num_val must be provided.")

    n_train = _to_abs(num_train)
    n_val = _to_abs(num_val)
    n_test = total_size - n_train - n_val if num_test is None else _to_abs(num_test)

    if any(n < 0 for n in (n_train, n_val, n_test)):
        raise ValueError("All split sizes must be non-negative.")

    if n_train + n_val + n_test != total_size:
        raise ValueError(
            f"train + val + test = {n_train}+{n_val}+{n_test} "
            f"!= total_size ({total_size})."
        )

    return n_train, n_val, n_test


def _normalize_proportions(
        proportions: Dict[str, float],
        dataset_names: List[str],
    ) -> Dict[str, float]:
    """
    Validate keys and normalise to sum=1.
    """
    missing = [n for n in dataset_names if n not in proportions]
    if missing:
        raise ValueError(f"Missing proportions for datasets: {missing}")

    total = float(sum(proportions.values()))
    if total <= 0:
        raise ValueError("Sum of proportions must be > 0.")

    return {k: float(v) / total for k, v in proportions.items()}


def _counts_from_proportions(
        split_size: int,
        proportions: Dict[str, float],
    ) -> Dict[str, int]:
    """
    Distribute split_size samples across datasets according to proportions.
    Uses floor + largest-remainder to guarantee exact sum.
    """
    names = list(proportions.keys())
    raw = {n: proportions[n] * split_size for n in names}
    base = {n: int(np.floor(raw[n])) for n in names}
    remainder = split_size - sum(base.values())

    if remainder > 0:
        order = sorted(names, key=lambda n: raw[n] - base[n], reverse=True)
        for i in range(remainder):
            base[order[i % len(order)]] += 1

    return base


def _assert_compatible_distance_units(datasets: Dict[str, ASEAtomsData]) -> str:
    """
    All component datasets must use the same distance unit.
    Returns the shared unit string.
    """
    units = {name: ds.distance_unit for name, ds in datasets.items()}
    unique = set(units.values())
    if len(unique) != 1:
        raise AtomsDataError(
            f"Component datasets have incompatible distance units: {units}. "
            "Convert them to the same unit before merging."
        )
    return unique.pop()


def _compute_merged_atomrefs(
        datasets: Dict[str, ASEAtomsData],
    ) -> Optional[Dict[str, torch.Tensor]]:
    """
    Return atomrefs only when every component dataset provides them AND
    they are numerically identical across all datasets.  Otherwise return None.
    """
    refs: List[Optional[Dict[str, torch.Tensor]]] = [
        getattr(ds, "atomrefs", None) for ds in datasets.values()
    ]

    if any(r is None for r in refs):
        return None

    base = refs[0]

    def _equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a.keys() != b.keys():
            return False
        for k in a:
            va, vb = a[k], b[k]
            if torch.is_tensor(va) and torch.is_tensor(vb):
                if va.shape != vb.shape or not torch.allclose(va, vb):
                    return False
            elif va != vb:
                return False
        return True

    for r in refs[1:]:
        if not _equal(base, r):
            return None

    return base


# ---------------------------------------------------------------------------
# PlanItem — lightweight record of one sample's origin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanItem:
    dataset_name: str  # key into MergedDataset.datasets
    index: int  # index inside that component dataset


# ---------------------------------------------------------------------------
# MergedDataset
# ---------------------------------------------------------------------------


class MergedDataset(Dataset):
    """
    Virtual merged dataset.

    Mirrors the ASEAtomsData public interface so it can be passed directly
    to AtomsDataModuleV2 as the ``dataset`` argument.

    Points To Discussed:
    --------------------
    * No merged DB is written to disk.
    * Splitting and proportional sampling happen at construction time via
      ``MergedDataset.from_datasets``.
    * Transforms are shared: a single list applied uniformly to every sample
      regardless of which component dataset it came from.
    * split-aware transform dispatch ("train" / "val" / "test") is supported
      through the ``split`` attribute, set automatically by ``subset()``.
    * ``dataset_id`` (int64 shape [1]) and ``source_index`` (int64 shape [1])
      are injected into every sample dict.
    * ``atomrefs`` are kept only when all component datasets agree exactly;
      otherwise None, so StatsAtomrefProvider will estimate them from data.
    """

    def __init__(
        self,
        datasets: Dict[str, ASEAtomsData],
        plan: List[PlanItem],
        atomrefs: Optional[Dict[str, torch.Tensor]],
        distance_unit: str,
        units: Dict[str, str],
        available_properties: List[str],
        add_source_index: bool = True,
    ) -> None:
        """
        Low-level constructor.  Prefer ``MergedDataset.from_datasets``.

        Args:
            datasets: component datasets keyed by name.
            plan: ordered list of (dataset_name, index) records.
            atomrefs: shared atomrefs or None.
            distance_unit: shared distance unit string.
            units: property → unit string dict (union of components).
            available_properties: union of component property names.
            add_source_index: inject ``source_index`` into each sample.
        """
        self.datasets = datasets
        self.plan = plan
        self.atomrefs = atomrefs
        self.distance_unit = distance_unit
        self._units = units
        self._available_properties = available_properties
        self.add_source_index = add_source_index

        # transform slots
        self.transforms: List[Transform] = []
        self.train_transforms: Optional[List[Transform]] = None
        self.val_transforms: Optional[List[Transform]] = None
        self.test_transforms: Optional[List[Transform]] = None

        # set by subset() to drive split-aware transform dispatch
        self.split: Optional[str] = None

        # AtomsDataModuleV2 reads these for its own _load_partitions path;
        # MergedDataset owns the split plan so we expose them as attributes.
        self._load_properties: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # ASEAtomsData-compatible properties
    # ------------------------------------------------------------------

    @property
    def available_properties(self) -> List[str]:
        return list(self._available_properties)

    @property
    def load_properties(self) -> List[str]:
        if self._load_properties is None:
            return self.available_properties
        return self._load_properties

    @load_properties.setter
    def load_properties(self, val: Optional[List[str]]) -> None:
        if val is not None:
            missing = [p for p in val if p not in self._available_properties]
            if missing:
                raise AtomsDataError(
                    f"Properties not available in merged dataset: {missing}"
                )
        self._load_properties = val

    @property
    def units(self) -> Dict[str, str]:
        return dict(self._units)

    # ------------------------------------------------------------------
    # subset() — mirrors ASEAtomsData.subset()
    # ------------------------------------------------------------------

    def subset(
        self,
        subset_idx: List[int],
        split: Optional[str] = None,
    ) -> "MergedDataset":
        """
        Return a shallow copy of this MergedDataset restricted to subset_idx
        rows of the current plan.

        This is the hook AtomsDataModuleV2 calls during setup() to build
        train / val / test views.
        """
        ds = copy.copy(self)
        ds.plan = [self.plan[i] for i in subset_idx]
        ds.split = split
        # transforms are intentionally shared by reference until DataModule
        # reassigns them via _initialize_transforms
        return ds

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item = self.plan[i]
        component_ds = self.datasets[item.dataset_name]

        # Pull the raw sample from the component dataset.
        # Component datasets must not apply their own transforms here because
        # shared transforms will be applied below.  We temporarily clear them.
        saved_transforms = component_ds.transforms
        saved_split = component_ds.split
        component_ds.transforms = []
        component_ds.split = None  # suppress split-aware dispatch in component
        try:
            sample = component_ds[item.index]
        finally:
            component_ds.transforms = saved_transforms
            component_ds.split = saved_split

        # Guarantee all tensors are at least 1D (collate uses torch.cat)
        for k, v in list(sample.items()):
            if torch.is_tensor(v) and v.dim() == 0:
                sample[k] = v.unsqueeze(0)

        # Inject provenance fields
        dataset_id = _dataset_name_to_id(item.dataset_name)
        sample["dataset_id"] = torch.tensor([dataset_id], dtype=torch.long)

        if self.add_source_index:
            sample["source_index"] = torch.tensor([item.index], dtype=torch.long)

        # Apply shared transforms (NeighborList, CastTo32, unit conversions …)
        for t in self._active_transforms():
            sample = t(sample)

        # Second pass to catch any scalar outputs from transforms
        for k, v in list(sample.items()):
            if torch.is_tensor(v) and v.dim() == 0:
                sample[k] = v.unsqueeze(0)

        return sample

    def _active_transforms(self) -> List[Transform]:
        """Return the correct transform list based on current split."""
        if self.split == "train" and self.train_transforms is not None:
            return self.train_transforms
        if self.split == "val" and self.val_transforms is not None:
            return self.val_transforms
        if self.split == "test" and self.test_transforms is not None:
            return self.test_transforms
        return self.transforms

    # ------------------------------------------------------------------
    # Factory: the main entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_datasets(
        cls,
        datasets: Dict[str, ASEAtomsData],
        proportions: Dict[str, float],
        total_size: int,
        num_train: SplitSize,
        num_val: SplitSize,
        num_test: Optional[SplitSize] = None,
        seed: int = 42,
        shuffle_within_split: bool = True,
        add_source_index: bool = True,
    ) -> "MergedDataset":
        """
        Build a single MergedDataset whose internal plan covers all three
        splits.  The three split index ranges are stored as ``num_train``,
        ``num_val``, ``num_test`` attributes so AtomsDataModuleV2 can call
        ``subset()`` on them.

        Sampling is without replacement within each component dataset and
        there is no leakage between splits.

        Args:
            datasets: component ASEAtomsData instances keyed by name.
            proportions: relative weight of each dataset in every split.
                Keys must match ``datasets``.  Values are normalised to sum=1.
            total_size: total number of samples across all splits.
            num_train: number (or fraction) of training samples.
            num_val: number (or fraction) of validation samples.
            num_test: number (or fraction) of test samples.
                If None, takes the remainder after train+val.
            seed: random seed for reproducible sampling.
            shuffle_within_split: shuffle the plan within each split.
            add_source_index: inject ``source_index`` into every sample.

        Returns:
            A MergedDataset whose plan is the concatenation of
            [train_plan | val_plan | test_plan].
            The split boundaries are stored as ``num_train``, ``num_val``,
            ``num_test`` integer attributes.
        """
        rng = np.random.default_rng(seed)
        dataset_names = list(datasets.keys())

        norm_props = _normalize_proportions(proportions, dataset_names)
        n_train, n_val, n_test = _resolve_split_sizes(
            total_size, num_train, num_val, num_test
        )

        train_counts = _counts_from_proportions(n_train, norm_props)
        val_counts = _counts_from_proportions(n_val, norm_props)
        test_counts = _counts_from_proportions(n_test, norm_props)

        # Validate availability (without-replacement constraint)
        for name in dataset_names:
            needed = train_counts[name] + val_counts[name] + test_counts[name]
            available = len(datasets[name])
            if needed > available:
                raise ValueError(
                    f"Not enough samples in '{name}': "
                    f"need {needed}, have {available}."
                )

        # Sample indices per component (without replacement, no leakage)
        train_plan: List[PlanItem] = []
        val_plan: List[PlanItem] = []
        test_plan: List[PlanItem] = []

        for name in dataset_names:
            ntr = train_counts[name]
            nva = val_counts[name]
            nts = test_counts[name]
            chosen = rng.choice(
                len(datasets[name]),
                size=ntr + nva + nts,
                replace=False,
            ).tolist()

            train_plan.extend(PlanItem(name, int(i)) for i in chosen[:ntr])
            val_plan.extend(PlanItem(name, int(i)) for i in chosen[ntr : ntr + nva])
            test_plan.extend(PlanItem(name, int(i)) for i in chosen[ntr + nva :])

        if shuffle_within_split:
            rng.shuffle(train_plan)
            rng.shuffle(val_plan)
            rng.shuffle(test_plan)

        # Shared metadata derived from components
        distance_unit = _assert_compatible_distance_units(datasets)
        atomrefs = _compute_merged_atomrefs(datasets)

        # Union of property names and units (property present in ANY component)
        all_props: Dict[str, str] = {}
        for ds in datasets.values():
            all_props.update(ds.units)

        # Concatenate plans: [train | val | test]
        full_plan = train_plan + val_plan + test_plan

        merged = cls(
            datasets=datasets,
            plan=full_plan,
            atomrefs=atomrefs,
            distance_unit=distance_unit,
            units=all_props,
            available_properties=list(all_props.keys()),
            add_source_index=add_source_index,
        )

        # Store split boundary counts so callers can drive AtomsDataModuleV2
        merged.num_train = n_train
        merged.num_val = n_val
        merged.num_test = n_test

        return merged


# ---------------------------------------------------------------------------
# dataset_id helper (avoids importing factories in every callsite)
# ---------------------------------------------------------------------------

# Simple deterministic mapping: hash the name to a stable int.
# Replace with your factories.DATASET_REGISTRY lookup if available.
_NAME_TO_ID: Dict[str, int] = {}
_NEXT_ID: int = 0


def _dataset_name_to_id(name: str) -> int:
    global _NEXT_ID
    if name not in _NAME_TO_ID:
        _NAME_TO_ID[name] = _NEXT_ID
        _NEXT_ID += 1
    return _NAME_TO_ID[name]

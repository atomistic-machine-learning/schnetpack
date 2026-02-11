from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from schnetpack.datasets.factories import get_dataset_id

__all__ = ["MergedDataset"]

SplitSize = Union[int, float]


def _resolve_split_sizes(
    total_size: int,
    num_train: SplitSize,
    num_val: SplitSize,
    num_test: Optional[SplitSize],
) -> Tuple[int, int, int]:
    """
    Convert split sizes to absolute integers.
    - If a split is a float <= 1, interpret as fraction of total_size.
    - If num_test is None, fill as remainder.
    """

    def to_abs(x):
        if isinstance(x, float) and x <= 1.0:
            return int(round(x * total_size))
        return int(x)

    if num_train is None or num_val is None:
        raise ValueError("num_train and num_val must be provided.")

    n_train = to_abs(num_train)
    n_val = to_abs(num_val)
    n_test = total_size - n_train - n_val if num_test is None else to_abs(num_test)

    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError("Split sizes must be non-negative.")

    if (n_train + n_val + n_test) != total_size:
        raise ValueError(
            f"train+val+test must equal total_size. "
            f"Got {n_train}+{n_val}+{n_test} != {total_size}"
        )

    return n_train, n_val, n_test


def _normalize_proportions(
    proportions: Dict[str, float], dataset_names: List[str]
) -> Dict[str, float]:
    """
    Ensure all dataset_names exist in proportions and normalize to sum=1.
    """
    for name in dataset_names:
        if name not in proportions:
            raise ValueError(f"Missing proportion for dataset '{name}'")

    s = float(sum(proportions.values()))
    if s <= 0:
        raise ValueError("Sum of proportions must be > 0")

    return {k: float(v) / s for k, v in proportions.items()}


def _counts_from_proportions(
    split_size: int, proportions: Dict[str, float]
) -> Dict[str, int]:
    """
    Turn proportions into integer counts that sum exactly to split_size.
    Uses floor then distributes remainder by largest fractional part.
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


@dataclass(frozen=True)
class PlanItem:
    dataset_name: str
    index: int


class MergedDataset(Dataset):
    """
    Virtual merged dataset:
      - Reads samples from underlying ASEAtomsData datasets
      - Injects:
          dataset_id: stable int (from factories.DATASET_REGISTRY) as shape (1,)
          source_index: original index in source dataset as shape (1,)
      - Applies transforms if provided (schnetpack transforms operate on dicts)

    Note: dataset_id/source_index MUST be at least 1D tensors (shape [1])
    because SchNetPack's collate uses torch.cat.
    """

    def __init__(
        self,
        datasets: Dict[str, Any],
        plan: List[PlanItem],
        add_source_index: bool = True,
        atomrefs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.datasets = datasets
        self.plan = plan
        self.add_source_index = add_source_index
        self.atomrefs = atomrefs

        # SchNetPack style: datasets expose .transforms list
        self.transforms: List[torch.nn.Module] = []

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item = self.plan[i]
        ds = self.datasets[item.dataset_name]

        sample = ds[item.index]  # dict of torch tensors (from ASEAtomsData)

        for k, v in list(sample.items()):
            if torch.is_tensor(v) and v.dim() == 0:
                sample[k] = v.view(1)

        did = get_dataset_id(item.dataset_name)
        sample["dataset_id"] = torch.tensor([did], dtype=torch.long)

        if self.add_source_index:
            sample["source_index"] = torch.tensor([int(item.index)], dtype=torch.long)

        # Apply transforms (NeighborList, CastTo32, etc.)
        for t in self.transforms:
            sample = t(sample)

        for k, v in list(sample.items()):
            if torch.is_tensor(v) and v.dim() == 0:
                sample[k] = v.view(1)

        return sample

    @staticmethod
    def make_splits(
        datasets: Dict[str, Any],
        proportions: Dict[str, float],
        total_size: int,
        num_train: SplitSize,
        num_val: SplitSize,
        num_test: Optional[SplitSize] = None,
        seed: int = 42,
        shuffle_within_split: bool = True,
        add_source_index: bool = True,
        atomrefs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple["MergedDataset", "MergedDataset", "MergedDataset"]:
        """
        Build train/val/test MergedDataset objects with:
          - dataset-level proportional counts in each split
          - without replacement across the entire split plan (no leakage)
          - shuffled ordering inside each split
        """
        rng = np.random.default_rng(seed)

        dataset_names = list(datasets.keys())
        proportions = _normalize_proportions(proportions, dataset_names)

        n_train, n_val, n_test = _resolve_split_sizes(
            total_size, num_train, num_val, num_test
        )

        train_counts = _counts_from_proportions(n_train, proportions)
        val_counts = _counts_from_proportions(n_val, proportions)
        test_counts = _counts_from_proportions(n_test, proportions)

        needed_total = {
            name: train_counts[name] + val_counts[name] + test_counts[name]
            for name in dataset_names
        }

        # Without replacement requires enough source samples
        for name in dataset_names:
            avail = len(datasets[name])
            need = needed_total[name]
            if need > avail:
                raise ValueError(
                    f"Not enough samples in '{name}' for without-replacement sampling. "
                    f"Need={need}, available={avail}."
                )

        train_plan: List[PlanItem] = []
        val_plan: List[PlanItem] = []
        test_plan: List[PlanItem] = []

        for name in dataset_names:
            chosen = rng.choice(
                len(datasets[name]), size=needed_total[name], replace=False
            ).tolist()
            ntr, nva, nts = train_counts[name], val_counts[name], test_counts[name]

            train_idx = chosen[:ntr]
            val_idx = chosen[ntr : ntr + nva]
            test_idx = chosen[ntr + nva : ntr + nva + nts]

            train_plan.extend([PlanItem(name, int(i)) for i in train_idx])
            val_plan.extend([PlanItem(name, int(i)) for i in val_idx])
            test_plan.extend([PlanItem(name, int(i)) for i in test_idx])

        if shuffle_within_split:
            rng.shuffle(train_plan)
            rng.shuffle(val_plan)
            rng.shuffle(test_plan)

        return (
            MergedDataset(
                datasets,
                train_plan,
                add_source_index=add_source_index,
                atomrefs=atomrefs,
            ),
            MergedDataset(
                datasets, val_plan, add_source_index=add_source_index, atomrefs=atomrefs
            ),
            MergedDataset(
                datasets,
                test_plan,
                add_source_index=add_source_index,
                atomrefs=atomrefs,
            ),
        )

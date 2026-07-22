from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import fasteners
import numpy as np
import torch

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.stats import calculate_stats, estimate_atomrefs

__all__ = ["StatsAtomrefProvider", "train_partition_fingerprint"]

# Same inter-process lock as split creation: statistics have the same lineage
# as the split, and sharing the lock means concurrent DDP ranks compute each
# entry at most once.
_SPLITTING_LOCK = "splitting.lock"


def train_partition_fingerprint(dataset_length: int, train_idx: List[int]) -> str:
    """
    Short deterministic fingerprint of a train partition.

    Statistics are a pure function of the dataset and the train partition;
    the fingerprint keys persisted statistics to that partition. The index
    list is sorted first — permutations of the same partition yield the same
    statistics.
    """
    payload = json.dumps(
        [int(dataset_length), sorted(int(i) for i in train_idx)]
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class StatsAtomrefProvider:
    """
    Compute and cache statistics and atom references of the training data.

    Statistics are a pure function of the base dataset and the train index
    list: the stats functions build their own raw (transform-free) view from
    the explicit indices, so the provider may be queried at any time — even
    after the dataset has its transforms attached — without ever computing
    on transformed data.

    With a stats file set, computed values are persisted to disk, keyed by
    the train-partition fingerprint: reruns and additional DDP ranks read
    instead of recompute. A None stats file disables persistence.
    """

    def __init__(
        self,
        dataset: ASEAtomsData,
        train_idx: List[int],
        stats_file: Optional[str] = None,
        fingerprint: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self.train_idx = list(train_idx)
        self.train_atomrefs = getattr(dataset, "atomrefs", None)
        self.stats_file = stats_file
        self.fingerprint = fingerprint or train_partition_fingerprint(
            len(dataset), self.train_idx
        )

        self._stats_cache: Dict[
            Tuple[str, bool, bool], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._atomref_cache: Dict[Tuple[str, bool], torch.Tensor] = {}

    # ---------- disk persistence ----------

    def _load_valid_entries(self) -> Dict[str, np.ndarray]:
        """Stored entries, or {} if absent or written for another partition."""
        if not os.path.exists(self.stats_file):
            return {}
        with np.load(self.stats_file) as data:
            entries = dict(data)
        if str(entries.pop("fingerprint", None)) != self.fingerprint:
            return {}
        return entries

    def _read_entry(self, entry_key: str) -> Optional[np.ndarray]:
        if self.stats_file is None:
            return None
        with fasteners.InterProcessLock(_SPLITTING_LOCK):
            return self._load_valid_entries().get(entry_key)

    def _write_entry(self, entry_key: str, value: np.ndarray) -> None:
        if self.stats_file is None:
            return
        with fasteners.InterProcessLock(_SPLITTING_LOCK):
            entries = self._load_valid_entries()
            entries[entry_key] = value
            np.savez(
                self.stats_file, fingerprint=np.array(self.fingerprint), **entries
            )

    # ---------- queries ----------

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats_cache:
            return self._stats_cache[key]

        entry_key = "stats:" + json.dumps(list(key))
        stored = self._read_entry(entry_key)
        if stored is not None:
            stats = (torch.tensor(stored[0]), torch.tensor(stored[1]))
            self._stats_cache[key] = stats
            return stats

        atomref = self.train_atomrefs if remove_atomref else None

        stats = calculate_stats(
            self.dataset,
            divide_by_atoms={property: divide_by_atoms},
            atomref=atomref,
            indices=self.train_idx,
        )[property]

        self._stats_cache[key] = stats
        mean, std = stats
        self._write_entry(
            entry_key, np.array([mean.item(), std.item()], dtype=np.float64)
        )
        return stats

    def get_atomrefs(
        self, property: str, is_extensive: bool, estimate: bool = True
    ) -> Dict[str, torch.Tensor]:
        # 1) If dataset already has atomrefs for this property, use them directly
        if self.train_atomrefs is not None and property in self.train_atomrefs:
            return {property: self.train_atomrefs[property]}

        if not estimate:
            raise RuntimeError(
                f"The dataset provides no atomrefs for property '{property}' "
                "and atomref estimation is disabled."
            )

        # 2) Otherwise estimate and cache. Only estimated atomrefs are
        # persisted — dataset-provided ones already live in the DB metadata.
        key = (property, is_extensive)
        if key in self._atomref_cache:
            return {property: self._atomref_cache[key]}

        entry_key = "atomrefs:" + json.dumps(list(key))
        stored = self._read_entry(entry_key)
        if stored is not None:
            atomref = torch.tensor(stored)
            self._atomref_cache[key] = atomref
            return {property: atomref}

        atomref = estimate_atomrefs(
            self.dataset,
            is_extensive={property: is_extensive},
            indices=self.train_idx,
        )[property]

        self._atomref_cache[key] = atomref
        self._write_entry(entry_key, atomref.detach().cpu().numpy())
        return {property: atomref}

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.stats import calculate_stats, estimate_atomrefs

__all__ = ["StatsAtomrefProvider"]


class StatsAtomrefProvider:
    """
    Compute and cache statistics and atom references of the training data.

    Statistics are a pure function of the base dataset and the train index
    list: the stats functions build their own raw (transform-free) view from
    the explicit indices, so the provider may be queried at any time — even
    after the dataset has its transforms attached — without ever computing
    on transformed data.
    """

    def __init__(self, dataset: ASEAtomsData, train_idx: List[int]) -> None:
        self.dataset = dataset
        self.train_idx = list(train_idx)
        self.train_atomrefs = getattr(dataset, "atomrefs", None)

        self._stats_cache: Dict[
            Tuple[str, bool, bool], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._atomref_cache: Dict[Tuple[str, bool], torch.Tensor] = {}

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats_cache:
            return self._stats_cache[key]

        atomref = self.train_atomrefs if remove_atomref else None

        stats = calculate_stats(
            self.dataset,
            divide_by_atoms={property: divide_by_atoms},
            atomref=atomref,
            indices=self.train_idx,
        )[property]

        self._stats_cache[key] = stats
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

        # 2) Otherwise estimate and cache
        key = (property, is_extensive)
        if key in self._atomref_cache:
            return {property: self._atomref_cache[key]}

        atomref = estimate_atomrefs(
            self.dataset,
            is_extensive={property: is_extensive},
            indices=self.train_idx,
        )[property]

        self._atomref_cache[key] = atomref
        return {property: atomref}

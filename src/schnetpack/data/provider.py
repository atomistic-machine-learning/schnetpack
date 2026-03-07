from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.stats import calculate_stats, estimate_atomrefs


class StatsAtomrefProvider:
    """
    Compute and cache statistics and atom references from the training dataset.
    """

    def __init__(self, train_dataset: ASEAtomsData) -> None:
        self.train_dataset = train_dataset
        self.train_atomrefs = getattr(train_dataset, "atomrefs", None)

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
            self.train_dataset,
            divide_by_atoms={property: divide_by_atoms},
            atomref=atomref,
        )[property]

        self._stats_cache[key] = stats
        return stats

    def get_atomrefs(
        self, property: str, is_extensive: bool
    ) -> Dict[str, torch.Tensor]:
        # 1) If dataset already has atomrefs for this property, use them directly
        if self.train_atomrefs is not None and property in self.train_atomrefs:
            return {property: self.train_atomrefs[property]}

        # 2) Otherwise estimate and cache
        key = (property, is_extensive)
        if key in self._atomref_cache:
            return {property: self._atomref_cache[key]}

        atomref = estimate_atomrefs(
            self.train_dataset,
            is_extensive={property: is_extensive},
        )[property]

        self._atomref_cache[key] = atomref
        return {property: atomref}

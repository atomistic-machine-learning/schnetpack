from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from schnetpack.data.loader import AtomsLoader
from schnetpack.data.stats import calculate_stats, estimate_atomrefs


@dataclass
class StatsAtomrefProvider:
    """
    Compute and cache statistics and atom references from the *training split*.
    """

    train_dataloader_factory: Callable[[], AtomsLoader]
    train_atomrefs: Optional[Dict[str, torch.Tensor]] = None

    _stats_cache: Optional[
        Dict[Tuple[str, bool, bool], Tuple[torch.Tensor, torch.Tensor]]
    ] = None
    _atomref_cache: Optional[Dict[Tuple[str, bool], torch.Tensor]] = None

    def __post_init__(self) -> None:
        if self._stats_cache is None:
            self._stats_cache = {}
        if self._atomref_cache is None:
            self._atomref_cache = {}

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            property: property key
            divide_by_atoms: if True, compute stats of property / n_atoms
            remove_atomref: if True, subtract atomref prior to stats computation
        Returns:
            (mean, std) tensors
        """
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats_cache:
            return self._stats_cache[key]

        loader = self.train_dataloader_factory()
        atomref = self.train_atomrefs if remove_atomref else None

        stats = calculate_stats(
            loader,
            divide_by_atoms={property: divide_by_atoms},
            atomref=atomref,
        )[property]

        self._stats_cache[key] = stats
        return stats

    def get_atomrefs(
        self, property: str, is_extensive: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            property: property key
            is_extensive: whether property is extensive
        Returns:
            dict {property: atomref_tensor}
        """
        key = (property, is_extensive)
        if key in self._atomref_cache:
            return {property: self._atomref_cache[key]}

        loader = self.train_dataloader_factory()
        atomref = estimate_atomrefs(loader, is_extensive={property: is_extensive})[
            property
        ]

        self._atomref_cache[key] = atomref
        return {property: atomref}

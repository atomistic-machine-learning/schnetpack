from typing import Dict, Tuple, Optional
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
        """
        Args:
            property: property name (e.g. "energy")
            divide_by_atoms: whether to divide the property by the number of atoms
            remove_atomref: whether to subtract the atomref from the property
        """
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
        """
        Args:
            property: property name (e.g. "energy")
            is_extensive: whether the property is extensive
        """
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


class MergedStatsAtomrefProvider:
    """
    Per-component stats provider for MergedDataset.
    """

    def __init__(self, train_dataset) -> None:
        """
        Args:
            train_dataset: a MergedDataset train split (produced by subset()).
                           Must have a plan and datasets attribute.
        """
        if not hasattr(train_dataset, "plan") or not hasattr(train_dataset, "datasets"):
            raise ValueError(
                "MergedStatsAtomrefProvider requires a MergedDataset instance "
                "with 'plan' and 'datasets' attributes."
            )

        self.train_dataset = train_dataset

        # Build one StatsAtomrefProvider per component dataset
        # using only that component's samples from the train split
        self.providers: Dict[str, StatsAtomrefProvider] = {}

        for name in train_dataset.datasets:
            # positions in train_dataset.plan that belong to this component
            component_idx = [
                i for i, (n, _) in enumerate(train_dataset.plan) if n == name
            ]

            if not component_idx:
                continue

            # subset of the train split containing only this component's samples
            component_train = train_dataset.subset(component_idx)

            self.providers[name] = StatsAtomrefProvider(component_train)

        # train_atomrefs per component — None if not available
        self.train_atomrefs: Dict[str, Optional[Dict[str, torch.Tensor]]] = {
            name: provider.train_atomrefs for name, provider in self.providers.items()
        }

    def get_stats(
        self,
        name: str,
        property: str,
        divide_by_atoms: bool,
        remove_atomref: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            name: component dataset name (e.g. "md17")
            property: property name (e.g. "energy")
            divide_by_atoms: whether to normalize by number of atoms
            remove_atomref: whether to subtract atomrefs before computing stats

        """
        if name not in self.providers:
            raise KeyError(
                f"No provider for dataset '{name}'. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[name].get_stats(property, divide_by_atoms, remove_atomref)

    def get_atomrefs(
        self,
        name: str,
        property: str,
        is_extensive: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            name: component dataset name (e.g. "md17")
            property: property name (e.g. "energy")
            is_extensive: whether the property is extensive
        """
        if name not in self.providers:
            raise KeyError(
                f"No provider for dataset '{name}'. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[name].get_atomrefs(property, is_extensive)

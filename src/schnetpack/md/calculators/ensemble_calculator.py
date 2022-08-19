from __future__ import annotations
import torch

from abc import ABC
from typing import TYPE_CHECKING, List, Dict, Optional
from schnetpack.md.calculators import MDCalculator

if TYPE_CHECKING:
    from schnetpack.md import System

__all__ = ["EnsembleCalculator"]


class EnsembleCalculator(ABC, MDCalculator):
    """
    Mixin for creating ensemble calculators from the standard `schnetpack.md.calculators` classes. Accumulates
    property predictions as the average over all models and uncertainties as the variance of model predictions.
    """

    def calculate(self, system: System):
        """
        Perform all calculations and compyte properties and uncertainties.

        Args:
            system (schnetpack.md.System): System from the molecular dynamics simulation.
        """
        inputs = self._generate_input(system)

        results = []
        for model in self.model:
            prediction = model(inputs)
            results.append(prediction)

        # Compute statistics
        self.results = self._accumulate_results(results)
        self._update_system(system)

    @staticmethod
    def _accumulate_results(
        results: List[Dict[str, torch.tensor]]
    ) -> Dict[str, torch.tensor]:
        """
        Accumulate results and compute average predictions and uncertainties.

        Args:
            results (list(dict(str, torch.tensor)):  List of output dictionaries of individual models.

        Returns:
            dict(str, torch.tensor): output dictionary with averaged predictions and uncertainties.
        """
        # Get the keys
        accumulated = {p: [] for p in results[0]}
        ensemble_results = {p: [] for p in results[0]}

        for p in accumulated:
            tmp = torch.stack([result[p] for result in results])
            ensemble_results[p] = torch.mean(tmp, dim=0)
            ensemble_results["{:s}_var".format(p)] = torch.var(tmp, dim=0)

        return ensemble_results

    def _activate_stress(self, stress_key: Optional[str] = None):
        """
        Routine for activating stress computations
        Args:
            stress_key (str, optional): stess label.
        """
        raise NotImplementedError

    def _update_required_properties(self):
        """
        Update required properties to also contain predictive variances.
        """
        new_required = []
        for p in self.required_properties:
            prop_string = "{:s}_var".format(p)
            new_required += [p, prop_string]
            # Update property conversion
            self.property_conversion[prop_string] = self.property_conversion[p]

        self.required_properties = new_required

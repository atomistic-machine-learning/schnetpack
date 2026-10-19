"""
Several models evaluated together, reported as one prediction per model.
"""

import torch
from torch import nn

__all__ = ["NNEnsemble"]


class NNEnsemble(nn.Module):
    """
    Several models evaluated together, reported as one prediction per model.

    An ensemble is itself a model: it takes a batch and returns a dictionary
    of predictions. What is different is that every entry carries a leading
    model axis, ``(n_models, ...)``, so the caller can take the mean, look at
    the spread, or both. Reducing that axis is left to the caller —
    :class:`~schnetpack.dynamics.EnsembleCalculator` and
    :class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`
    average it, and hand the unreduced stack to an
    :class:`~schnetpack.uncertainty.Uncertainty`.

    The models it holds run on the same batch, so they share its units and
    its neighbor list.
    """

    def __init__(self, models: nn.ModuleList, properties: str | list[str]):
        """
        Args:
            models: the models of the ensemble
            properties: names of the properties to collect from them; every
                model must predict every one
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        if isinstance(properties, str):
            properties = [properties]
        self.properties = list(properties)

    def setup(self, stage: str | None = None) -> None:
        for model in self.models:
            setup = getattr(model, "setup", None)
            if setup is not None:
                setup(stage)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        collected: dict[str, list[torch.Tensor]] = {p: [] for p in self.properties}
        for index, model in enumerate(self.models):
            predictions = model(dict(inputs))  # shallow copy of inputs
            missing = [prop for prop in self.properties if prop not in predictions]
            if missing:
                raise KeyError(
                    f"model {index} of the ensemble predicts no {missing}; "
                    f"got {sorted(predictions)}"
                )
            for prop in self.properties:
                collected[prop].append(predictions[prop])
        return {prop: torch.stack(values) for prop, values in collected.items()}

from typing import Dict, Optional

import torch
from ase.data import atomic_masses

import schnetpack.properties as structure
from .base import Transform
from schnetpack.nn import scatter_add

__all__ = [
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
    "AddOffsets",
    "RemoveOffsets",
]


class SubtractCenterOfMass(Transform):
    """
    Subtract center of mass from positions.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        masses = torch.tensor(atomic_masses[inputs[structure.Z]])
        inputs[structure.position] -= (
            masses.unsqueeze(-1) * inputs[structure.position]
        ).sum(0) / masses.sum()
        return inputs


class SubtractCenterOfGeometry(Transform):
    """
    Subtract center of geometry from positions.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs[structure.position] -= inputs[structure.position].mean(0)
        return inputs


class RemoveOffsets(Transform):
    """
    Remove offsets from property based on the mean of the training data and/or the single atom reference calculations.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(
        self,
        property,
        remove_mean: bool = False,
        remove_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
    ):
        super().__init__()
        self._property = property
        self.remove_mean = remove_mean
        self.remove_atomrefs = remove_atomrefs
        self.is_extensive = is_extensive

        assert (
            remove_atomrefs or remove_mean
        ), "You should set at least one of `remove_mean` and `remove_atomrefs` to true!"

        if self.remove_atomrefs:
            self.register_buffer("atomref", torch.zeros((zmax,)))
        if self.remove_mean:
            self.register_buffer("mean", torch.zeros((1,)))

    def datamodule(self, value):
        self._datamodule = value

        if self.remove_atomrefs:
            atrefs = self._datamodule.train_dataset.atomrefs
            self.atomref = atrefs[self._property].detach()

        if self.remove_mean:
            stats = self._datamodule.get_stats(
                self._property, self.is_extensive, self.remove_atomrefs
            )
            self.mean = stats[0].detach()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.remove_mean:
            inputs[self._property] -= self.mean * inputs[structure.n_atoms]

        if self.remove_atomrefs:
            inputs[self._property] -= torch.sum(self.atomref[inputs[structure.Z]])

        return inputs


class AddOffsets(Transform):
    """
    Add offsets to property based on the mean of the training data and/or the single atom reference calculations.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True
    atomref: torch.Tensor

    def __init__(
        self,
        property,
        add_mean: bool = False,
        add_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
    ):
        super().__init__()
        self._property = property
        self.add_mean = add_mean
        self.add_atomrefs = add_atomrefs
        self.is_extensive = is_extensive
        self._aggregation = "sum" if self.is_extensive else "mean"

        assert (
            add_mean or add_atomrefs
        ), "You should set at least one of `add_mean` and `add_atomrefs` to true!"

        self.register_buffer("atomref", torch.zeros((zmax,)))
        self.register_buffer("mean", torch.zeros((1,)))

    def datamodule(self, value):
        if self.add_atomrefs:
            atrefs = value.train_dataset.atomrefs
            self.atomref = atrefs[self._property].detach()

        if self.add_mean:
            stats = value.get_stats(
                self._property, self.is_extensive, self.add_atomrefs
            )
            self.mean = stats[0].detach()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.add_mean:
            inputs[self._property] += self.mean * inputs[structure.n_atoms]

        if self.add_atomrefs:
            idx_m = inputs[structure.idx_m]
            y0i = self.atomref[inputs[structure.Z]]
            maxm = int(idx_m[-1]) + 1

            y0 = scatter_add(y0i, idx_m, dim_size=maxm)

            if not self.is_extensive:
                y0 /= inputs[structure.n_atoms]

            inputs[self._property] -= y0

        return inputs

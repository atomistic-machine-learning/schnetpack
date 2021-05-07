from typing import Dict, Optional

import torch
from ase.data import atomic_masses

import schnetpack as spk
import schnetpack.structure as structure
from .transform import Transform

__all__ = [
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
    "UnitConversion",
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
        results: Optional[Dict[str, torch.Tensor]] = None,
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
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        inputs[structure.position] -= inputs[structure.position].mean(0)
        return inputs


class UnitConversion(Transform):
    """
    Convert units of selected properties.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(self, property_unit_dict: Dict[str, str]):
        """
        Args:
            property_unit_dict: mapping property name to target unit,
                specified as a string (.e.g. 'kcal/mol').
        """
        self.property_unit_dict = property_unit_dict
        self.src_units = None
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x = inputs if self.mode == "pre" else results

        if not self.src_units:
            units = self.data.units
            self.src_units = {p: units[p] for p in self.property_unit_dict}

        for prop, tgt_unit in self.property_unit_dict.items():
            x[prop] *= spk.units.convert_units(self.src_units[prop], tgt_unit)
        return x


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

    @Transform.datamodule.setter
    def datamodule(self, value):
        self._datamodule = value

        if self.remove_atomrefs:
            atrefs = self._datamodule.train_dataset.atomrefs
            self.atomref.copy_(torch.tensor(atrefs[self._property]))

        if self.remove_mean:
            stats = self._datamodule.get_stats(
                self._property, self.is_extensive, self.remove_atomrefs
            )
            self.mean.copy_(stats[0])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x = inputs if self.mode == "pre" else results

        if self.remove_mean:
            x[self._property] -= self.mean * inputs[structure.n_atoms]

        if self.remove_atomrefs:
            x[self._property] -= torch.sum(self.atomref[inputs[structure.Z]])

        return x


class AddOffsets(Transform):
    """
    Add offsets to property based on the mean of the training data and/or the single atom reference calculations.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

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

        if self.add_atomrefs:
            self.register_buffer("atomref", torch.zeros((zmax,)))
        if self.add_mean:
            self.register_buffer("mean", torch.zeros((1,)))

    @Transform.datamodule.setter
    def datamodule(self, value):
        self._datamodule = value

        if self.add_atomrefs:
            atrefs = self._datamodule.train_dataset.atomrefs
            self.atomref.copy_(torch.tensor(atrefs[self._property]))

        if self.add_mean:
            stats = self._datamodule.get_stats(
                self._property, self.is_extensive, self.remove_atomrefs
            )
            self.mean.copy_(stats[0])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x = inputs if self.mode == "pre" else results

        if self.add_mean:
            x[self._property] += self.mean * inputs[structure.n_atoms]

        if self.add_atomrefs:
            idx_m = inputs[structure.idx_m]
            y0i = self.atomref[inputs[structure.Z]]
            tmp = torch.zeros(
                (idx_m[-1] + 1, self.n_out), dtype=y0i.dtype, device=y0i.device
            )
            y0 = tmp.index_add(0, idx_m, y0i)
            if not self.is_extensive:
                y0 /= input[structure.n_atoms]
            x[self._property] -= y0

        return x

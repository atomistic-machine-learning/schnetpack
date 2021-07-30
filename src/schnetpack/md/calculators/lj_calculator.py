from __future__ import annotations
from typing import Union, Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md.neighborlist_md import NeighborListMD

import torch
import torch.nn as nn

from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.neighborlist_md import ASENeighborListMD

from schnetpack import properties
import schnetpack.nn as snn
from schnetpack.atomistic import Forces

__all__ = ["LJCalculator", "LJModel"]


class LJCalculator(SchnetPackCalculator):
    def __init__(
        self,
        r_equilibrium: float,
        well_depth: float,
        force_label: str,
        energy_units: Union[str, float],
        position_units: Union[str, float],
        energy_label: str = None,
        stress_label: str = None,
        property_conversion: Dict[str, Union[str, float]] = {},
        neighbor_list: NeighborListMD = ASENeighborListMD,
        cutoff: float = 5.0,
        healing_length: float = 3.0,
        cutoff_shell: float = 1.0,
    ):
        model = LJModel(
            r_equilibrium=r_equilibrium,
            well_depth=well_depth,
            cutoff=cutoff,
            healing_length=healing_length,
            calc_forces=True,
            calc_stress=(stress_label is not None),
            energy_key=energy_label,
            force_key=force_label,
            stress_key=stress_label,
        )

        super(LJCalculator, self).__init__(
            model,
            force_label=force_label,
            energy_units=energy_units,
            position_units=position_units,
            energy_label=energy_label,
            stress_label=stress_label,
            property_conversion=property_conversion,
            neighbor_list=neighbor_list,
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
        )

    def _load_model(self, model_file):
        return model_file

    def _init_neighbor_list(
        self, neighbor_list: NeighborListMD, cutoff: float, cutoff_shell: float
    ):
        # Check if atom triples need to be computed (e.g. for Behler functions)
        return neighbor_list(
            cutoff=cutoff, cutoff_shell=cutoff_shell, requires_triples=False
        )


class LJModel(nn.Module):
    def __init__(
        self,
        r_equilibrium: float,
        well_depth: float,
        cutoff: float,
        healing_length: float,
        calc_forces: bool = True,
        calc_stress: bool = True,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        stress_key: str = properties.stress,
    ):
        super(LJModel, self).__init__()

        self.r_equilibrium = r_equilibrium
        self.well_depth = well_depth

        self.calc_forces = calc_forces
        self.calc_stress = calc_stress
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        self.cutoff = CustomCutoff(cutoff_radius=cutoff, healing_length=healing_length)

        self.force_layer = Forces(
            calc_forces=calc_forces,
            calc_stress=calc_stress,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
        )

    def forward(self, inputs):
        # Activate gradient for force/stress computations
        if self.calc_forces or self.calc_stress:
            inputs[properties.Rij].requires_grad_()

        vec_ij = inputs[properties.Rij]
        positions = inputs[properties.R]
        idx_i = inputs[properties.idx_i]
        r_ij = torch.norm(vec_ij, dim=1, keepdim=True)
        r_cut = self.cutoff(r_ij)

        # Compute lennard jones potential
        power_6 = torch.pow(self.r_equilibrium / r_ij, 6)
        power_12 = power_6 * power_6

        yij = (power_12 - power_6) * r_cut

        # aggregate
        yi = snn.scatter_add(yij, idx_i, dim_size=positions.shape[0])

        idx_m = inputs[properties.idx_m]

        maxm = int(idx_m[-1]) + 1
        tmp = torch.zeros((maxm, 1), dtype=yi.dtype, device=yi.device)
        y = tmp.index_add(0, idx_m, yi)
        y = torch.squeeze(y, -1)

        y = self.well_depth * 0.5 * y

        # collect results
        inputs[self.energy_key] = y

        # Compute forces and stress
        inputs.update(self.force_layer(inputs))

        results = {self.energy_key: y}

        if self.calc_forces:
            results[self.force_key] = inputs[self.force_key].detach()

        if self.calc_stress:
            results[self.stress_key] = inputs[self.stress_key].detach()

        return results


class CustomCutoff(nn.Module):
    def __init__(self, cutoff_radius: float, healing_length: float):
        super(CustomCutoff, self).__init__()
        self.register_buffer("cutoff_radius", torch.Tensor([cutoff_radius]))
        self.register_buffer("healing_length", torch.Tensor([healing_length]))

    def forward(self, distances):
        r = (
            distances - (self.cutoff_radius - self.healing_length)
        ) / self.healing_length
        r_function = 1.0 + r ** 2 * (2.0 * r - 3.0)

        switch = torch.where(
            distances > self.cutoff_radius - self.healing_length,
            r_function,
            torch.ones_like(distances),
        )
        switch = torch.where(
            distances > self.cutoff_radius, torch.zeros_like(distances), switch
        )

        return switch

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict

if TYPE_CHECKING:
    from schnetpack.md.neighborlist_md import NeighborListMD

import torch
import torch.nn as nn

from schnetpack.md.calculators import SchNetPackCalculator

from schnetpack import properties
import schnetpack.nn as snn
from schnetpack.atomistic import Forces, PairwiseDistances, Strain

__all__ = ["LJCalculator", "LJModel"]


class LJCalculator(SchNetPackCalculator):
    """
    Lennard Jones potential calculator. Primarily used for testing barostats and thermostats.

    Args:
        r_equilibrium (float): equilibrium distance in position units
        well_depth (float): depth of the Lennard-Jones potential in energy units.
        force_key (str): String indicating the entry corresponding to the molecular forces
        energy_unit (float, float): Conversion factor converting the energies returned by the used model back to
                                     internal MD units.
        position_unit (float, float): Conversion factor for converting the system positions to the units required by
                                       the model.
        neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                    interatomic distances should be computed.
        energy_key (str, optional): If provided, label is used to store the energies returned by the model to the
                                      system.
        stress_key (str, optional): If provided, label is used to store the stress returned by the model to the
                                      system (required for constant pressure simulations).
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        healing_length (float): Healing length used for the cutoff potential.
    """

    def __init__(
        self,
        r_equilibrium: float,
        well_depth: float,
        force_key: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        neighbor_list: NeighborListMD,
        energy_key: str = None,
        stress_key: str = None,
        property_conversion: Dict[str, Union[str, float]] = {},
        healing_length: float = 3.0,
    ):
        model = LJModel(
            r_equilibrium=r_equilibrium,
            well_depth=well_depth,
            cutoff=neighbor_list.cutoff,
            healing_length=healing_length,
            calc_forces=True,
            calc_stress=(stress_key is not None),
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
        )

        super(LJCalculator, self).__init__(
            model,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            stress_key=stress_key,
            property_conversion=property_conversion,
        )

    def _prepare_model(self, model_file: LJModel):
        """
        Dummy routine, since no model has to be loaded.

        Args:
            model_file (LJModel): Initialized Lennard-Jones model.

        Returns:
            LJModel: input model.
        """
        return model_file


class LJModel(nn.Module):
    """
    Lennard Jones potential calculator. Primarily used for testing barostats and thermostats.

    Args:
        r_equilibrium (float): equilibrium distance in position units
        well_depth (float): depth of the Lennard-Jones potential in energy units.
        cutoff (float): Cutoff radius for computing the neighbor interactions. If this is set to a negative number,
                        the cutoff is determined automatically based on the model (default=-1.0). Units are the distance
                        units used in the model.
        healing_length (float): Healing length used for the cutoff potential.
        calc_forces (bool): toggle force computation.
        calc_stress (bool): toggle stress computation.
        energy_key (str): Key used for storing energies.
        force_key (str): Key used for storing forces.
        stress_key (str): Key used for storing stress.
    """

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

        self.cutoff_function = CustomCutoff(
            cutoff_radius=cutoff, healing_length=healing_length
        )

        # Modules for distances, stress tensor and forces
        self.distances = PairwiseDistances()
        self.strain = Strain()
        self.force_layer = Forces(
            calc_forces=calc_forces,
            calc_stress=calc_stress,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the Lennard-Jones energy and forces if requested.

        Args:
            inputs (dict(str, torch.Tensor)):  Input dictionary.

        Returns:
            dict(str, torch.Tensor):  Dictionary of model outputs.
        """

        # Activate gradient for force/stress computations
        if self.calc_forces:
            inputs[properties.R].requires_grad_()
        if self.calc_stress:
            inputs = self.strain(inputs)

        # Compute interatomic distances
        inputs = self.distances(inputs)

        vec_ij = inputs[properties.Rij]
        positions = inputs[properties.R]
        idx_i = inputs[properties.idx_i]
        r_ij = torch.norm(vec_ij, dim=1, keepdim=True)
        r_cut = self.cutoff_function(r_ij)

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
        """
        Custom cutoff for Lennard-Jones potentials using a healing length.

        Args:
            cutoff_radius (float): cutoff radius.
            healing_length (float): healing length.
        """
        super(CustomCutoff, self).__init__()
        self.register_buffer("cutoff_radius", torch.Tensor([cutoff_radius]))
        self.register_buffer("healing_length", torch.Tensor([healing_length]))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute cutoff based on the input distances.

        Args:
            distances (torch.tensor):

        Returns:
            torch.tensor: cutoff function applied to the distances.
        """
        # Compute basic component
        r = (
            distances - (self.cutoff_radius - self.healing_length)
        ) / self.healing_length
        r_function = 1.0 + r**2 * (2.0 * r - 3.0)

        # Compute second part of cutoff
        switch = torch.where(
            distances > self.cutoff_radius - self.healing_length,
            r_function,
            torch.ones_like(distances),
        )
        # Compute third component
        switch = torch.where(
            distances > self.cutoff_radius, torch.zeros_like(distances), switch
        )

        return switch

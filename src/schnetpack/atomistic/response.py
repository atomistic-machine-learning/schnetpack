from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch.autograd import grad

import schnetpack.properties as properties

__all__ = ["Forces", "StrainResponse"]


class ResponseException(Exception):
    pass


class Forces(nn.Module):
    """
    Predicts forces and stress as response of the energy prediction
    w.r.t. the atom positions.

    """

    def __init__(
        self,
        calc_forces: bool = True,
        calc_stress: bool = False,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        stress_key: str = properties.stress,
    ):
        """
        Args:
            calc_forces: If True, calculate atomic forces.
            calc_stress: If True, calculate the stress tensor.
            energy_key: Key of the energy in results.
            force_key: Key of the forces in results.
            stress_key: Key of the stress in results.
        """
        super(Forces, self).__init__()
        self.calc_forces = calc_forces
        self.calc_stress = calc_stress
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        self.required_derivatives = []
        if self.calc_forces:
            self.required_derivatives.append(properties.Rij)
        if self.calc_stress:
            self.required_derivatives.append(properties.strain)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Epred = inputs[self.energy_key]
        R = inputs[properties.position]
        results = {}

        go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
        grads = grad(
            [Epred],
            [inputs[prop] for prop in self.required_derivatives],
            grad_outputs=go,
            create_graph=self.training,
        )

        if self.calc_forces:
            dEdRij = grads[0]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdRij is None:
                dEdRij = torch.zeros_like(inputs[properties.Rij])

            Fpred = torch.zeros_like(R)
            Fpred = Fpred.index_add(0, inputs[properties.idx_i], dEdRij)
            Fpred = Fpred.index_add(0, inputs[properties.idx_j], -dEdRij)
            results[self.force_key] = Fpred

        if self.calc_stress:
            stress = grads[-1]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if stress is None:
                stress = torch.zeros_like(inputs[properties.cell])

            cell = inputs[properties.cell]
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[:, :, None]
            results[self.stress_key] = stress  # / volume

        return results


class StrainResponse(nn.Module):
    """
    THis is required to calculate the stress as a response property.
    Adds strain-dependence to relative atomic positions Rij and (optionally) to absolute positions and unit cell.
    """

    def __init__(
        self, strain_Rij: bool = True, strain_R: bool = False, strain_cell: bool = False
    ):
        super().__init__()
        self.strain_Rij = strain_Rij
        self.strain_R = strain_R
        self.strain_cell = strain_cell

    def forward(self, inputs: Dict[str, torch.Tensor]):
        idx_m = inputs[properties.idx_m]
        idx_i = inputs[properties.idx_i]
        strain = torch.zeros_like(inputs[properties.cell])
        strain.requires_grad_()
        inputs[properties.strain] = strain

        strain_i = strain[idx_m]
        if self.strain_Rij:
            strain_ij = strain_i[idx_i]
            inputs[properties.Rij] = inputs[properties.Rij] + torch.bmm(
                strain_ij, inputs[properties.Rij][:, :, None]
            ).squeeze(-1)

        if self.strain_R:
            inputs[properties.R] = inputs[properties.R] + torch.sum(
                inputs[properties.R][:, :, None] * strain_i, dim=1
            )

        if self.strain_cell:
            inputs[properties.cell] = inputs[properties.cell] + torch.sum(
                inputs[properties.cell][:, :, :, None] * strain[:, None, :, :], dim=2
            )

        return inputs

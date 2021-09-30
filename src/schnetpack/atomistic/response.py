from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch.autograd import grad

import schnetpack.properties as properties

__all__ = ["Forces"]


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
            self.required_derivatives.append(properties.R)
        if self.calc_stress:
            self.required_derivatives.append(properties.cell)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Epred = inputs[self.energy_key]
        results = {}

        go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
        dx = []
        if self.calc_forces:
            dx.append(inputs[properties.R])
        if self.calc_stress:
            dx.append(inputs[properties.cell])

        grads = grad([Epred], dx, grad_outputs=go, create_graph=self.training,)

        if self.calc_forces:
            dEdR = grads[0]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdR is None:
                dEdR = torch.zeros_like(inputs[properties.Rij])
            results[self.force_key] = -dEdR

        if self.calc_stress:
            dEdC = grads[-1]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdC is None:
                dEdC = torch.zeros_like(inputs[properties.cell])

            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            cell_33 = inputs[properties.cell].view(maxm, 3, 3)
            volume = (
                torch.sum(
                    cell_33[:, 0, :]
                    * torch.cross(cell_33[:, 1, :], cell_33[:, 2, :], dim=1),
                    dim=1,
                    keepdim=True,
                )
                .expand(maxm, 3)
                .reshape(maxm * 3, 1)
            )
            results[self.stress_key] = dEdC / volume

        return results


class PositionGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, cell, Rij, idx_i, idx_j, idx_m):
        ctx.Rshape = R.shape

        if cell.requires_grad:
            ctx.save_for_backward(idx_i, idx_j, Rij, cell, idx_m)
        else:
            ctx.save_for_backward(idx_i, idx_j)
        return Rij

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        idx_i, idx_j = inputs[:2]

        if ctx.needs_input_grad[0]:
            dR = torch.zeros(
                ctx.Rshape, dtype=grad_output.dtype, device=grad_output.device
            )
            dR = dR.index_add(0, idx_i, -grad_output)
            dR = dR.index_add(0, idx_j, grad_output)
        else:
            dR = None

        if ctx.needs_input_grad[1]:
            Rij, cell, idx_m = inputs[2:]

            dcell_i = torch.zeros(
                (ctx.Rshape[0], 3, 3),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            # sum over j
            dcell_i = dcell_i.index_add(
                0, idx_i, grad_output[:, None, :] * Rij[:, :, None],
            )
            maxm = int(idx_m[-1]) + 1
            dcell = torch.zeros(
                (maxm, 3, 3), dtype=dcell_i.dtype, device=dcell_i.device
            )
            dcell = dcell.index_add(0, idx_m, dcell_i)
            dcell = dcell.reshape(maxm * 3, 3)
        else:
            dcell = None

        return dR, dcell, grad_output, None, None, None


position_grad = PositionGrad.apply

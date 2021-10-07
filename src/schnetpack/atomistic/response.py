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
            self.required_derivatives.append(properties.strain)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Epred = inputs[self.energy_key]
        results = {}

        go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
        dx = []
        if self.calc_forces:
            dx.append(inputs[properties.R])
        if self.calc_stress or True:
            dx.append(inputs[properties.strain])

        grads = grad(
            [Epred],
            dx,
            grad_outputs=go,
            create_graph=self.training,
        )

        if self.calc_forces:
            dEdR = grads[0]
            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdR is None:
                dEdR = torch.zeros_like(inputs[properties.Rij])
            results[self.force_key] = -dEdR

        # if self.calc_stress:
        #     dEdC = grads[-1]
        #     # TorchScript needs Tensor instead of Optional[Tensor]
        #     if dEdC is None:
        #         dEdC = torch.zeros_like(inputs[properties.strain])
        #
        #     idx_m = inputs[properties.idx_m]
        #     maxm = int(idx_m[-1]) + 1
        #     cell_33 = inputs[properties.cell]  # .view(maxm, 3, 3)
        #     volume = (
        #         torch.sum(
        #             cell_33[:, 0, :]
        #             * torch.cross(cell_33[:, 1, :], cell_33[:, 2, :], dim=1),
        #             dim=1,
        #             keepdim=True,
        #         )
        #         .expand(maxm, 3)
        #         .reshape(maxm * 3, 1)
        #     )
        #     # TODO: colume shape, etc
        #     results[self.stress_key] = dEdC  # / volume

        return results


class RijGrads(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        R: torch.Tensor,
        Rij: torch.Tensor,
        strain: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        idx_m: torch.Tensor,
    ):
        ctx.Rshape = R.shape

        if strain.requires_grad:
            ctx.save_for_backward(idx_i, idx_j, R, Rij, idx_m)
        else:
            ctx.save_for_backward(idx_i, idx_j)
        return Rij

    @staticmethod
    def backward(ctx, go_Rij: torch.Tensor):
        inputs = ctx.saved_tensors
        idx_i, idx_j = inputs[:2]

        if ctx.needs_input_grad[0]:
            # dRij/dR
            dR = torch.zeros(ctx.Rshape, dtype=go_Rij.dtype, device=go_Rij.device)
            dR = dR.index_add(0, idx_i, -go_Rij)
            dR = dR.index_add(0, idx_j, go_Rij)
        else:
            dR = None

        if ctx.needs_input_grad[2]:
            R, Rij, idx_m = inputs[2:]

            dstrain_i = torch.zeros(
                (ctx.Rshape[0], 3, 3),
                dtype=go_Rij.dtype,
                device=go_Rij.device,
            )
            dstrain_i = dstrain_i.index_add(
                0,
                idx_i,
                go_Rij[:, None, :] * Rij[:, :, None],
            )

            # sum over i
            maxm = int(idx_m[-1]) + 1
            dstrain = torch.zeros(
                (maxm, 3, 3), dtype=dstrain_i.dtype, device=dstrain_i.device
            )
            dstrain = dstrain.index_add(0, idx_m, dstrain_i)
        else:
            dstrain = None

        return (
            dR,
            go_Rij,
            dstrain,
            None,
            None,
            None,
        )


class StrainPositions(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        R: torch.Tensor,
        strain: torch.Tensor,
        idx_m: torch.Tensor,
    ):
        ctx.Rshape = R.shape
        ctx.save_for_backward(R, idx_m)
        return R

    @staticmethod
    def backward(ctx, go_R: torch.Tensor):
        R, idx_m = ctx.saved_tensors

        dstrain_i = torch.zeros(
            (ctx.Rshape[0], 3, 3),
            dtype=go_R.dtype,
            device=go_R.device,
        )
        dstrain_i += go_R[:, None, :] * R[:, :, None]

        # sum over i
        maxm = int(idx_m[-1]) + 1
        dstrain = torch.zeros(
            (maxm, 3, 3), dtype=dstrain_i.dtype, device=dstrain_i.device
        )
        dstrain = dstrain.index_add(0, idx_m, dstrain_i)

        return (
            go_R,
            dstrain,
            None,
        )


class StrainCell(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cell: torch.Tensor,
        strain: torch.Tensor,
    ):
        ctx.save_for_backward(cell)
        return cell

    @staticmethod
    def backward(ctx, go_cell: torch.Tensor):
        cell = ctx.saved_tensors[0]
        dstrain = torch.matmul(go_cell, cell.transpose(1, 2))
        return None, dstrain


def setup_input_grads(inputs: Dict[str, torch.Tensor], required_derivatives):
    required_derivatives += [properties.strain]

    if properties.strain in required_derivatives:
        inputs[properties.strain] = torch.zeros_like(inputs[properties.cell])

    for p in required_derivatives:
        inputs[p].requires_grad_()

    inputs[properties.Rij] = RijGrads.apply(
        inputs[properties.R],
        inputs[properties.Rij],
        inputs[properties.strain],
        inputs[properties.idx_i],
        inputs[properties.idx_j],
        inputs[properties.idx_m],
    )

    inputs = strain_structure(inputs)
    return inputs


def strain_structure(inputs: Dict[str, torch.Tensor]):
    strain = inputs[properties.strain]

    inputs[properties.R_strained] = StrainPositions.apply(
        inputs[properties.R], strain, inputs[properties.idx_m]
    )
    inputs[properties.cell_strained] = StrainCell.apply(inputs[properties.cell], strain)
    return inputs

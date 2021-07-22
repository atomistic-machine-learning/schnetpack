from typing import Sequence, Union, Callable, Dict, Optional, List

import torch
import torch.nn as nn

import schnetpack as spk
import schnetpack.properties as properties
from torch.autograd import grad

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

        self.required_derivatives = [properties.Rij]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Epred = inputs[self.energy_key]
        R = inputs[properties.position]
        results = {}

        go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
        dEdRij = grad(
            [Epred],
            [inputs[properties.Rij]],
            grad_outputs=go,
            create_graph=self.training,
        )[0]

        # TorchScript needs Tensor instead of Optional[Tensor]
        if dEdRij is None:
            dEdRij = torch.zeros_like(inputs[properties.Rij])

        if self.calc_forces is not None:
            Fpred = torch.zeros_like(R)
            Fpred = Fpred.index_add(0, inputs[properties.idx_i], dEdRij)
            Fpred = Fpred.index_add(0, inputs[properties.idx_j], -dEdRij)
            results[self.force_key] = Fpred

        if self.calc_stress is not None:
            stress_i = torch.zeros((R.shape[0], 3, 3), dtype=R.dtype, device=R.device)

            # sum over j
            stress_i = stress_i.index_add(
                0,
                inputs[properties.idx_i],
                dEdRij[:, None, :] * inputs[properties.Rij][:, :, None],
            )

            # sum over i
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            stress = torch.zeros(
                (maxm, 3, 3), dtype=stress_i.dtype, device=stress_i.device
            )
            stress = stress.index_add(0, idx_m, stress_i)

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
            results[self.stress_key] = stress.reshape(maxm * 3, 3) / volume

        return results

import torch
import torch.nn as nn
from typing import Union, Callable, Dict, Optional

from schnetpack import properties
import schnetpack.units as spk_units


__all__ = ["CovalentBond"]


class CovalentBond(nn.Module):
    """
    This module computes a harmonic oscillator potential between two atoms.

    Warning: This module requires consistent atom indexing across all configurations. Furthermore, all configurations
    are required to have equal number of atoms.

    """

    def __init__(
        self,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        output_key: str,
        trainable: bool = False,
        atom_tuple: tuple = (100, 187),
        bond_length=2.366,
        spring_konst=0.5,
    ):

        super(CovalentBond, self).__init__()

        self.idx0 = atom_tuple[0]
        self.idx1 = atom_tuple[1]

        self.energy_units = spk_units.convert_units("eV", energy_unit)
        self.position_units = spk_units.convert_units("Ang", position_unit)

        self.output_key = output_key

        self.bond_length = nn.Parameter(
            torch.tensor([bond_length]), requires_grad=trainable
        )
        self.spring_konst = nn.Parameter(
            torch.tensor([spring_konst]), requires_grad=trainable
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        n_structures = len(inputs["_n_atoms"])
        n_atoms = inputs["_n_atoms"][0].item()

        # check if n_atoms is identical for all configs
        if not (inputs["_n_atoms"] - n_atoms).sum().item() == 0:
            raise Warning("n atoms is not identical for all configs")
        index_offsets = torch.tensor(
            [_ for _ in range(0, n_structures * n_atoms, n_atoms)]
        )

        indices0 = self.idx0 + index_offsets
        indices1 = self.idx1 + index_offsets

        # define harmonic oscillator
        pos = inputs[properties.R] * self.position_units
        deviation = self.bond_length - torch.norm(pos[indices0] - pos[indices1], dim=-1)
        harm_pot = self.spring_konst * deviation**2

        inputs[self.output_key] = harm_pot / self.energy_units

        return inputs

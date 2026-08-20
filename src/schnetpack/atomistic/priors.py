import torch
import torch.nn as nn
from typing import Dict, Tuple

import schnetpack.properties as properties
import schnetpack.units as spk_units

__all__ = ["HarmonicBond"]


class HarmonicBond(nn.Module):
    """
    A harmonic restraint on the distance between two atoms of every structure.

    Follows the shape of schnetpack's other priors, e.g. ``ZBLRepulsionEnergy``: the
    module contributes its own energy term under ``output_key`` and leaves the summing
    to an ``Aggregation`` module, rather than writing into the model's energy directly.

    The restraint is stated in eV and Angstrom and converted to whatever units the model
    works in. ``atom_pair`` indexes into a single structure, so the two atoms have to
    carry the same indices in every structure of the batch.

    Args:
        atom_pair: indices, within a single structure, of the two restrained atoms.
        bond_length: equilibrium distance of the restraint, in Angstrom.
        force_constant: spring constant of the restraint, in eV/Angstrom**2.
        energy_unit (str/float): Energy unit the model works in.
        position_unit (str/float): Unit used for distances by the model.
        output_key (str): Key to which results will be stored.
        trainable (bool): If set to true, bond length and force constant will be
            optimized during training (default=False).
    """

    def __init__(
        self,
        atom_pair: Tuple[int, int],
        bond_length: float,
        force_constant: float,
        energy_unit: str,
        position_unit: str,
        output_key: str,
        trainable: bool = False,
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]

        self.register_buffer("atom_pair", torch.tensor(atom_pair, dtype=torch.long))
        self.to_angstrom = spk_units.convert_units(position_unit, "Ang")
        self.to_model_energy = spk_units.convert_units("eV", energy_unit)

        self.bond_length = nn.Parameter(
            torch.tensor(bond_length), requires_grad=trainable
        )
        self.force_constant = nn.Parameter(
            torch.tensor(force_constant), requires_grad=trainable
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # the first atom of every structure, so that structures of differing size work
        n_atoms = inputs[properties.n_atoms]
        first = torch.cumsum(n_atoms, dim=0) - n_atoms

        positions = inputs[properties.R]
        Rij = (
            positions[first + self.atom_pair[1]] - positions[first + self.atom_pair[0]]
        )
        distance = torch.norm(Rij, dim=-1) * self.to_angstrom

        energy = self.force_constant * (distance - self.bond_length) ** 2
        inputs[self.output_key] = energy * self.to_model_energy

        return inputs

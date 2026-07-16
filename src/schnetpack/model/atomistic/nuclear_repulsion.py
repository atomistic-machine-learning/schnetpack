import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Dict, Optional

import schnetpack.properties as properties
import schnetpack.nn as snn
import schnetpack.units as spk_units

__all__ = ["ZBLRepulsionEnergy"]


class ZBLRepulsionEnergy(nn.Module):
    """
    Computes a Ziegler-Biersack-Littmark style repulsion energy

    Args:
        energy_unit (str/float): Energy unit.
        position_unit (str/float): Unit used for distances.
        output_key (str): Key to which results will be stored
        trainable (bool): If set to true, ZBL parameters will be optimized during training (default=True)
        cutoff_fn (Callable): Apply a cutoff function to the interatomic distances.

    References:
    .. [#Cutoff] Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
       Texturing & Modeling: A Procedural Approach;
       Morgan Kaufmann, 2003
    .. [#ZBL]
       https://docs.lammps.org/pair_zbl.html
    """

    def __init__(
        self,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        output_key: str,
        trainable: bool = True,
        cutoff_fn: Optional[Callable] = None,
    ):
        super(ZBLRepulsionEnergy, self).__init__()

        energy_units = spk_units.convert_units("Ha", energy_unit)
        position_units = spk_units.convert_units("Bohr", position_unit)
        ke = energy_units * position_units
        self.register_buffer("ke", torch.tensor(ke))

        self.cutoff_fn = cutoff_fn
        self.output_key = output_key

        # Basic ZBL parameters (in atomic units)
        # Since all quantities have a predefined sign, they are initialized to the inverse softplus and a softplus
        # function is applied in the forward pass to guarantee the correct sign during training
        a_div = snn.softplus_inverse(
            torch.tensor([1.0 / (position_units * 0.8854)])
        )  # in this way, distances can be used directly
        a_pow = snn.softplus_inverse(torch.tensor([0.23]))
        exponents = snn.softplus_inverse(
            torch.tensor([3.19980, 0.94229, 0.40290, 0.20162])
        )
        coefficients = snn.softplus_inverse(
            torch.tensor([0.18175, 0.50986, 0.28022, 0.02817])
        )

        # Initialize network parameters
        self.a_pow = nn.Parameter(a_pow, requires_grad=trainable)
        self.a_div = nn.Parameter(a_div, requires_grad=trainable)
        self.coefficients = nn.Parameter(coefficients, requires_grad=trainable)
        self.exponents = nn.Parameter(exponents, requires_grad=trainable)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1)
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        idx_m = inputs[properties.idx_m]

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        # Construct screening function
        a = z ** F.softplus(self.a_pow)
        a_ij = (a[idx_i] + a[idx_j]) * F.softplus(self.a_div)
        # Get exponents and coefficients, normalize the latter
        exponents = a_ij[..., None] * F.softplus(self.exponents)[None, ...]
        coefficients = F.softplus(self.coefficients)[None, ...]
        coefficients = F.normalize(coefficients, p=1.0, dim=1)

        screening = torch.sum(
            coefficients * torch.exp(-exponents * d_ij[:, None]), dim=1
        )

        # Compute nuclear repulsion
        repulsion = (z[idx_i] * z[idx_j]) / d_ij

        # Apply cutoff if requested
        if self.cutoff_fn is not None:
            f_cut = self.cutoff_fn(d_ij)
            repulsion = repulsion * f_cut

        # Compute ZBL energy
        y_zbl = snn.scatter_add(repulsion * screening, idx_i, dim_size=n_atoms)
        y_zbl = snn.scatter_add(y_zbl, idx_m, dim_size=n_molecules)
        y_zbl = 0.5 * self.ke * y_zbl

        inputs[self.output_key] = y_zbl

        return inputs

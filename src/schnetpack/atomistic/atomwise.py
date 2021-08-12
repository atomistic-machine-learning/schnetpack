from typing import Sequence, Union, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.nn as snn
import schnetpack.properties as properties


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        """
        Args:
            output_key: the key under which the result will be stored
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.output_key = output_key
        self.per_atom_output_key = per_atom_output_key

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.MLP(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # aggregate
        if self.aggregation_mode is not None:
            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms][:, None]

            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            tmp = torch.zeros((maxm, self.n_out), dtype=y.dtype, device=y.device)
            y = tmp.index_add(0, idx_m, y)
            y = torch.squeeze(y, -1)

        return {self.output_key: y}


class DipoleMoment(nn.Module):
    """
    Predicts dipole moments from latent partial charges and (optionally) local, atomic dipoles.
    The latter requires a representation supplying (equivariant) vector features.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        predict_magnitude: bool = False,
        return_charges: bool = False,
        dipole_key: str = properties.dipole_moments,
        charges_key: str = properties.partial_charges,
        use_vector_representation: bool = False,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            predict_magnitude: If true, calculate magnitude of dipole
            use_vector_representation: If true, use vector representation to predict local,
                atomic dipoles
        """
        super().__init__()

        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.return_charges = return_charges
        self.predict_magnitude = predict_magnitude
        self.use_vector_representation = use_vector_representation

        if use_vector_representation:
            layers = [
                snn.GatedEquivariantBlock(
                    n_sin=n_in,
                    n_vin=n_in,
                    n_sout=n_in,
                    n_vout=n_hidden,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=activation,
                )
                for _ in range(n_layers - 1)
            ]
            layers.append(
                snn.GatedEquivariantBlock(
                    n_sin=n_hidden,
                    n_vin=n_hidden,
                    n_sout=1,
                    n_vout=1,
                    n_hidden=n_hidden,
                    activation=activation,
                )
            )
            self.outnet = nn.Sequential(*layers)
        else:
            self.outnet = spk.nn.MLP(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["representation"]
        result = {}

        if self.use_vector_representation:
            l1 = inputs["vector_representation"]
            charges, atomic_dipoles = self.outnet((l0, l1))
            atomic_dipoles = torch.squeeze(atomic_dipoles, -1)
        else:
            charges = self.outnet(l0)
            atomic_dipoles = 0.0

        if self.return_charges:
            result[self.charges_key] = charges

        y = positions * charges
        if self.use_vector_representation:
            y = y + atomic_dipoles

        # sum over atoms
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        tmp = torch.zeros((maxm, self.n_out), dtype=y.dtype, device=y.device)
        y = tmp.index_add(0, idx_m, y)
        y = torch.squeeze(y, -1)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=True)

        result[self.dipole_key] = y
        return result

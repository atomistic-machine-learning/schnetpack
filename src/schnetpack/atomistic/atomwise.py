from typing import Sequence, Union, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.properties as structure
import schnetpack.nn as snn

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
        custom_outnet: Callable = None,
        module_dim = False,
        outnet_input: Union[str, Sequence[str]] = "scalar_representation",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
        physnet_energy: bool = False
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
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.n_out = n_out
        
        self.outnet_input = outnet_input
        self.output_key = output_key
        self.per_atom_output_key = per_atom_output_key

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            weight_init = True
        )
        
        self.aggregation_mode = aggregation_mode
        
        self.module_dim = module_dim
        self.physnet_energy = physnet_energy
        
#         if module_dim:
#             self.outnet.weight.data = torch.nn.Parameter(torch.zeros(n_out,n_in))
#             self.outnet.bias.data.fill_(0.)
            

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        if self.module_dim:
            inputs[self.outnet_input] = inputs[self.outnet_input].sum(0)
        
        if self.aggregation_mode == "avg":
            yi = yi / inputs[structure.n_atoms][:, None]

        y = self.outnet(inputs["scalar_representation"])
        
        if self.physnet_energy:
            y += inputs[properties.physnet_aggregate]

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

    References:
    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#irspec] Gastegger, Behler, Marquetand.
       Machine learning molecular dynamics for the simulation of infrared spectra.
       Chemical science 8.10 (2017): 6924-6935.
    .. [#dipole] Veit et al.
       Predicting molecular dipole moments by combining atomic partial charges and atomic dipoles.
       The Journal of Chemical Physics 153.2 (2020): 024113.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        predict_magnitude: bool = False,
        return_charges: bool = False,
        dipole_key: str = properties.dipole_moment,
        charges_key: str = properties.partial_charges,
        use_vector_representation: bool = False,
        module_dim = False,
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
            return_charges: If true, return latent partial charges
            dipole_key: the key under which the dipoles will be stored
            charges_key: the key under which partial charges will be stored
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
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                weight_init=True
            )
        self.module_dim = module_dim

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        result = {}
        
        if self.module_dim:
            l0 = l0.sum(0)

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
        tmp = torch.zeros((maxm, 3), dtype=y.dtype, device=y.device)
        y = tmp.index_add(0, idx_m, y)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=False)

        result[self.dipole_key] = y
        return result


class Polarizability(nn.Module):
    """
    Predicts polarizability tensor using tensor rank factorization.
    This requires an equivariant representation, e.g. PaiNN, that provides both scalar and vectorial features.

    References:
    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        predict_isotropic: bool = False,
        polarizability_key: str = properties.polarizability,
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
            predict_isotropic: If true, return isotropic polarizability
            polarizability_key: the key under which the predicted polarizability will be stored
        """
        super(Polarizability, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.polarizability_key = polarizability_key
        self.predict_isotropic = predict_isotropic

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]
        dim = l1.shape[-2]

        l0, l1 = self.outnet((l0, l1))

        # isotropic on diagonal
        alpha = l0[..., 0:1]
        size = list(alpha.shape)
        size[-1] = dim
        alpha = alpha.expand(*size)
        alpha = torch.diag_embed(alpha)

        # add anisotropic components
        mur = l1[..., None, 0] * positions[..., None, :]
        alpha_c = mur + mur.transpose(-2, -1)
        alpha = alpha + alpha_c

        # sum over atoms
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        tmp = torch.zeros((maxm, 3, 3), dtype=alpha.dtype, device=alpha.device)
        alpha = tmp.index_add(0, idx_m, alpha)

        if self.predict_isotropic:
            alpha = torch.einsum("bii->b", alpha) / 3.0

        result = {self.polarizability_key: alpha}
        return result

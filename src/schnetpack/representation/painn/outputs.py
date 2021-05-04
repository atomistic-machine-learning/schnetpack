import torch
from torch import nn
import torch.nn.functional as F

import schnetpack.nn as snn
from schnetpack import Properties


class GatedEquivariantBlock(nn.Module):
    """
    The gated equivariant block is used to obtain rotationally invariant and equivariant features to be used
    for tensorial prop
    """

    def __init__(
        self,
        n_sin,
        n_vin,
        n_sout,
        n_vout,
        n_hidden,
        activation=F.silu,
        sactivation=None,
    ):
        """
        Args:
            n_sin: input dimension of scalar features
            n_vin: input dimension of vectorial features
            n_sout: output dimension of scalar features
            n_vout: output dimension of vectorial features
            n_hidden: size of hidden layers
            activation: activation of hidden layers
            sactivation: final activation to scalar features
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = snn.Dense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            snn.Dense(n_sin + n_vout, n_hidden, activation=activation),
            snn.Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars, vectors):
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


class Dipole(nn.Module):
    """ Output layer for dipole moment """

    def __init__(
        self,
        n_in,
        n_hidden,
        activation=F.silu,
        property="dipole",
        predict_magnitude=False,
    ):
        """
        Args:
            n_in: input dimension of atomwise features
            n_hidden: size of hidden layers
            activation: activation function
            property: name of property to be predicted
            predict_magnitude: If true, calculate magnitude of dipole
        """
        super().__init__()

        self.property = property
        self.derivative = None
        self.predict_magnitude = predict_magnitude

        self.equivariant_layers = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    n_sin=n_in,
                    n_vin=n_in,
                    n_sout=n_in,
                    n_vout=n_hidden,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=activation,
                ),
                GatedEquivariantBlock(
                    n_sin=n_hidden,
                    n_vin=n_hidden,
                    n_sout=1,
                    n_vout=1,
                    n_hidden=n_hidden,
                    activation=activation,
                ),
            ]
        )
        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs):
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask]
        l0 = inputs["representation"]
        l1 = inputs["vector_representation"]

        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

        atomic_dipoles = torch.squeeze(l1, -1)
        charges = l0
        dipole_offsets = positions * charges

        y = atomic_dipoles + dipole_offsets
        y = y * atom_mask[..., None]
        y = torch.sum(y, dim=1)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=True)

        result = {self.property: y}
        return result


class Polarizability(nn.Module):
    """ Output layer for polarizability tensor """

    def __init__(
        self,
        n_in,
        n_hidden,
        activation=F.silu,
        property="polarizability",
    ):
        """
        Args:
            n_in: input dimension of atomwise features
            n_hidden: size of hidden layers
            activation: activation function
            property: name of property to be predicted
        """
        super(Polarizability, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.property = property
        self.derivative = None

        self.equivariant_layers = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    n_sin=n_in,
                    n_vin=n_in,
                    n_sout=n_hidden,
                    n_vout=n_hidden,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=activation,
                ),
                GatedEquivariantBlock(
                    n_sin=n_in,
                    n_vin=n_hidden,
                    n_sout=1,
                    n_vout=1,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=None,
                ),
            ]
        )

        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs):
        atom_mask = inputs[Properties.atom_mask]
        positions = inputs[Properties.R]
        l0 = inputs["representation"]
        l1 = inputs["vector_representation"]
        dim = l1.shape[-2]

        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

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
        alpha = alpha * atom_mask[..., None, None]
        alpha = torch.sum(alpha, dim=1)

        result = {self.property: alpha}
        return result

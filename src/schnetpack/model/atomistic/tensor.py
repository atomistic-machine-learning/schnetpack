from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
from schnetpack import properties
from schnetpack.model_format import register
from schnetpack.nn.so3 import rsh_l2_to_matrix

__all__ = ["AtomwiseTensor", "PairwiseTensor"]


@register
class AtomwiseTensor(nn.Module):
    """
    Predicts a symmetric 3x3 tensor per structure, e.g. a stress or a strain, as
    the mean over the atoms of

        T_i = a_i 1 + matrix(b_i),

    with a scalar a_i from the atom's scalar features and the five components
    b_i of a symmetric traceless matrix from its l=2 features, whose channels
    are weighted by scalars computed from the scalar features. The tensor turns
    with the structure.

    This requires a representation with l=2 features, e.g.
    :class:`~schnetpack.model.SO3net` with ``lmax >= 2``. On a site with an
    inversion center the l=1 features vanish, the l=2 features do not.

    The last layer starts at zero, so an untrained head returns zero.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int | Sequence[int] | None = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = "y",
    ):
        """
        Args:
            n_in: number of features of the representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            output_key: the key under which the tensor is stored
        """
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        # a_i and one weight per channel of the l=2 features
        self.outnet = snn.build_mlp(
            n_in, n_in + 1, n_hidden, n_layers, activation, last_zero_init=True
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = inputs["multipole_representation"]
        if x.shape[1] < 9:
            raise ValueError("AtomwiseTensor needs l=2 features (lmax >= 2).")
        out = self.outnet(x[:, 0])
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        l2 = (x[:, 4:9] * out[:, None, 1:]).sum(-1)
        y = out[:, :1, None] * eye + rsh_l2_to_matrix(l2)

        n_atoms = inputs[properties.n_atoms]
        y = snn.scatter_add(y, inputs[properties.idx_m], dim_size=n_atoms.shape[0])
        # clamp: a padded batch may hold empty structures (no atoms)
        inputs[self.output_key] = y / n_atoms.clamp(min=1).to(y.dtype)[:, None, None]
        return inputs


@register
class PairwiseTensor(nn.Module):
    """
    Predicts a symmetric 3x3 tensor per structure, e.g. a stress or a strain, as
    the mean over the atoms of a sum over each atom's neighbors,

        T = 1/N sum_i sum_j c_ij d_ij d_ij^T,

    with the unit vector d_ij between the two atoms and a scalar c_ij from their
    scalar features and their distance. The tensor turns with the structure.

    Only scalar features are read, so any representation serves. The pair
    directions carry the shear: an atom's vector features vanish on an inversion
    center (rocksalt, fcc), in a strained cell too.

    The last layer starts at zero, so an untrained head returns zero.
    """

    def __init__(
        self,
        n_in: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_hidden: int | None = None,
        activation: Callable = F.silu,
        output_key: str = "y",
    ):
        """
        Args:
            n_in: number of features of the scalar representation
            radial_basis: expands the distances; its ``n_rbf`` is read
            cutoff_fn: weight of a pair, zero at the cutoff of the neighbor list
            n_hidden: number of pair features (default: n_in)
            activation: activation of the atoms' features
            output_key: the key under which the tensor is stored
        """
        super().__init__()
        n_hidden = n_hidden or n_in
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.output_key = output_key
        self.model_outputs = [output_key]

        self.neighbor_net = snn.Dense(n_in, n_hidden, activation=activation)
        self.filter_net = snn.Dense(radial_basis.n_rbf, n_hidden, activation=None)
        self.center_net = snn.Dense(n_in, n_hidden, activation=None)
        nn.init.zeros_(self.center_net.weight)
        nn.init.zeros_(self.center_net.bias)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = inputs["scalar_representation"]
        idx_i, idx_j = inputs[properties.idx_i], inputs[properties.idx_j]
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1)
        dir_ij = r_ij / (d_ij[:, None] + 1e-8)
        fcut = self.cutoff_fn(d_ij)[:, None]
        filters = self.filter_net(self.radial_basis(d_ij)) * fcut
        center, neighbor = self.center_net(h)[idx_i], self.neighbor_net(h)[idx_j]
        c_ij = (center * filters * neighbor).sum(-1)
        y = c_ij[:, None, None] * dir_ij[:, :, None] * dir_ij[:, None, :]

        n_atoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        y = snn.scatter_add(y, idx_m[idx_i], dim_size=n_atoms.shape[0])
        # clamp: a padded batch may hold empty structures (no atoms)
        inputs[self.output_key] = y / n_atoms.clamp(min=1).to(y.dtype)[:, None, None]
        return inputs

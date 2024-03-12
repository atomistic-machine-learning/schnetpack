from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
import schnetpack.nn.so3 as so3
import schnetpack.properties as properties
from schnetpack.nn import ElectronicEmbedding

__all__ = ["SO3net"]


class SO3net(nn.Module):
    """
    A simple SO3-equivariant representation using spherical harmonics and
    Clebsch-Gordon tensor products.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        lmax: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        shared_interactions: bool = False,
        max_z: int = 101,
        return_vector_representation: bool = False,
        activation: Optional[Callable] = F.silu,
        activate_charge_spin_embedding: bool = False,
        nuclear_embedding: Union[Callable, nn.Module] = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            lmax: maximum angular momentum of spherical harmonics basis
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            shared_interactions:
            max_z: maximal nuclear charge
            return_vector_representation: return l=1 features in Cartesian XYZ order
                (e.g. for DipoleMoment output module)
            activate_charge_spin_embedding: if True, charge and spin embeddings are added to nuclear embeddings taken from SpookyNet Implementation
            nuclear_embedding: type of nuclear embedding to use (simple is simple embedding and complex is the one with electron configuration)
        """
        super(SO3net, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.lmax = lmax
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis
        self.return_vector_representation = return_vector_representation
        self.nuclear_embedding = nuclear_embedding
        self.activate_charge_spin_embedding = activate_charge_spin_embedding
        self.activation = activation

        if self.nuclear_embedding is None:
            self.nuclear_embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        # needed if spin or charge embeeding requested
        if self.activate_charge_spin_embedding:

            # additional embeedings for the total charge
            self.charge_embedding = ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=True)
            # additional embeedings for the spin multiplicity
            self.magmom_embedding = ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=False) 

        self.sphharm = so3.RealSphericalHarmonics(lmax=lmax)

        self.so3convs = snn.replicate_module(
            lambda: so3.SO3Convolution(lmax, n_atom_basis, self.radial_basis.n_rbf),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings1 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings2 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings3 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.gatings = snn.replicate_module(
            lambda: so3.SO3ParametricGatedNonlinearity(n_atom_basis, lmax),
            self.n_interactions,
            shared_interactions,
        )
        self.so3product = so3.SO3TensorProduct(lmax)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        # inputs needed for charge and spin embedding
        num_batch = len(inputs[properties.idx])
        batch_seg = inputs[properties.idx_m]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij

        Yij = self.sphharm(dir_ij)
        radial_ij = self.radial_basis(d_ij)
        cutoff_ij = self.cutoff_fn(d_ij)[..., None]

        x0 = self.nuclear_embedding(atomic_numbers)[:, None]

        if self.activate_charge_spin_embedding:
            # get inputs for spin and charge embedding 
            # to avoid error not having total charge /spin multiplicity in db if embedding not used
            total_charge = inputs[properties.total_charge]
            spin = inputs[properties.spin_multiplicity]

            # specific total charge embeeding - squeezing necessary to remove the extra dimension, which is not anticipated in forward pass of electronic embedding
            charge_embedding = self.charge_embedding(x0.squeeze(),total_charge,num_batch,batch_seg)[:,None]

            # specific spin embeeding
            spin_embedding = self.magmom_embedding(x0.squeeze(),spin,num_batch,batch_seg)[:,None]

            # additive combining nuclear, charge and spin embeeding
            x0 = (x0 + charge_embedding + spin_embedding)

        x = so3.scalar2rsh(x0, int(self.lmax))

        for so3conv, mixing1, mixing2, gating, mixing3 in zip(
                self.so3convs, self.mixings1, self.mixings2, self.gatings, self.mixings3
        ):
            dx = so3conv(x, radial_ij, Yij, cutoff_ij, idx_i, idx_j)
            ddx = mixing1(dx)
            dx = dx + self.so3product(dx, ddx)
            dx = mixing2(dx)
            dx = gating(dx)
            dx = mixing3(dx)
            x = x + dx

        inputs["scalar_representation"] = x[:, 0]
        inputs["multipole_representation"] = x

        # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
        if self.return_vector_representation:
            inputs["vector_representation"] = torch.roll(x[:, 1:4], 1, 1)

        return inputs

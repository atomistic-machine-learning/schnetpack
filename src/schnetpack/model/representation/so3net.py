from typing import Callable, Dict, Optional, Union, List

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
        return_vector_representation: bool = False,
        activation: Optional[Callable] = F.silu,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
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
            return_vector_representation: return l=1 features in Cartesian XYZ order
                (e.g. for DipoleMoment output module)
            nuclear_embedding: custom nuclear embedding (e.g. spk.nn.embeddings.NuclearEmbedding)
            electronic_embeddings: list of electronic embeddings. E.g. for spin and
                charge (see spk.nn.embeddings.ElectronicEmbedding)
        """
        super(SO3net, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.lmax = lmax
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis
        self.return_vector_representation = return_vector_representation
        self.activation = activation

        # initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding
        if electronic_embeddings is None:
            electronic_embeddings = []
        electronic_embeddings = nn.ModuleList(electronic_embeddings)
        self.electronic_embeddings = electronic_embeddings

        # initialize shperical harmonics
        self.sphharm = so3.RealSphericalHarmonics(lmax=lmax)

        # initialize filters
        self.so3convs = snn.replicate_module(
            lambda: so3.SO3Convolution(lmax, n_atom_basis, self.radial_basis.n_rbf),
            self.n_interactions,
            shared_interactions,
        )

        # initialize interaction blocks
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

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij

        Yij = self.sphharm(dir_ij)
        radial_ij = self.radial_basis(d_ij)
        cutoff_ij = self.cutoff_fn(d_ij)[..., None]

        # compute initial embeddings
        x0 = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            x0 = x0 + embedding(x0, inputs)
        x0 = x0.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
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

        # collect results
        inputs["scalar_representation"] = x[:, 0]
        inputs["multipole_representation"] = x
        # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
        if self.return_vector_representation:
            inputs["vector_representation"] = torch.roll(x[:, 1:4], 1, 1)

        return inputs

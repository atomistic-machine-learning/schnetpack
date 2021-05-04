import math
import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from schnetpack import Properties
from schnetpack.nn.neighbors import atom_distances
from typing import Union, Callable


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        a = self.freqs[None, None, None, :]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm

        return y


class PaiNN(nn.Module):
    """ Polarizable atom interaction neural network """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        cutoff_network: Union[nn.Module, str] = "cosine",
        radial_basis: Callable = BesselBasis,
        activation=F.silu,
        max_z: int = 100,
        store_neighbors: bool = False,
        store_embeddings: bool = False,
    ):
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.cutoff_network = snn.get_cutoff_by_string(cutoff_network)(cutoff)
        self.radial_basis = radial_basis(cutoff=cutoff, n_rbf=n_rbf)
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        self.store_neighbors = store_neighbors
        self.store_embeddings = store_embeddings

        if type(activation) is str:
            if activation == "swish":
                activation = F.silu
            elif activation == "softplus":
                activation = snn.shifted_softplus

        self.filter_net = snn.Dense(
            n_rbf, self.n_interactions * 3 * n_atom_basis, activation=None
        )

        self.interatomic_context_net = nn.ModuleList(
            [
                nn.Sequential(
                    snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
                    snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
                )
                for _ in range(self.n_interactions)
            ]
        )

        self.intraatomic_context_net = nn.ModuleList(
            [
                nn.Sequential(
                    snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
                    snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
                )
                for _ in range(self.n_interactions)
            ]
        )

        self.mu_channel_mix = nn.ModuleList(
            [
                snn.Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)
                for _ in range(self.n_interactions)
            ]
        )

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]

        # get interatomic vectors and distances
        rij, dir_ij = atom_distances(
            positions=positions,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            cell=cell,
            cell_offsets=cell_offset,
            return_vecs=True,
            normalize_vecs=True,
        )

        phi_ij = self.radial_basis(rij[..., None])
        fcut = self.cutoff_network(rij) * neighbor_mask
        fcut = fcut.unsqueeze(-1)

        if self.store_neighbors:
            inputs["distances"] = rij
            inputs["directions"] = dir_ij
            inputs["rbf_ij"] = phi_ij
            inputs["fcut"] = fcut

        filters = self.filter_net(phi_ij) * fcut
        filters = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # initialize scalar and vector embeddings
        scalars = self.embedding(atomic_numbers)
        if self.store_embeddings:
            inputs["atom_embedding"] = scalars

        sshape = scalars.shape
        vectors = torch.zeros(
            (sshape[0], sshape[1], 3, sshape[2]), device=scalars.device
        )

        for i in range(self.n_interactions):
            # message function
            h_i = self.interatomic_context_net[i](scalars)
            h_j, vectors_j = self.collect_neighbors(h_i, vectors, neighbors)

            # neighborhood context
            h_i = filters[i] * h_j

            dscalars, dvR, dvv = torch.split(h_i, self.n_atom_basis, dim=-1)
            dvectors = torch.einsum("bijf,bijd->bidf", dvR, dir_ij) + torch.einsum(
                "bijf,bijdf->bidf", dvv, vectors_j
            )
            dscalars = torch.sum(dscalars, dim=2)
            scalars = scalars + dscalars
            vectors = vectors + dvectors

            # update function
            mu_mix = self.mu_channel_mix[i](vectors)
            vectors_V, vectors_U = torch.split(mu_mix, self.n_atom_basis, dim=-1)
            mu_Vn = torch.norm(vectors_V, dim=2)

            ctx = torch.cat([scalars, mu_Vn], dim=-1)
            h_i = self.intraatomic_context_net[i](ctx)
            ds, dv, dsv = torch.split(h_i, self.n_atom_basis, dim=-1)
            dv = dv.unsqueeze(2) * vectors_U
            dsv = dsv * torch.einsum("bidf,bidf->bif", vectors_V, vectors_U)

            # calculate atomwise updates
            scalars = scalars + ds + dsv
            vectors = vectors + dv

        inputs["vector_representation"] = vectors
        return scalars

    def collect_neighbors(self, scalars, vectors, neighbors):
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)

        scalar_nbh = nbh.expand(-1, -1, scalars.size(2))
        scalars_j = torch.gather(scalars, 1, scalar_nbh)
        scalars_j = scalars_j.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        vectors_nbh = nbh[..., None].expand(-1, -1, vectors.size(2), vectors.size(3))
        vectors_j = torch.gather(vectors, 1, vectors_nbh)
        vectors_j = vectors_j.view(nbh_size[0], nbh_size[1], nbh_size[2], 3, -1)
        return scalars_j, vectors_j

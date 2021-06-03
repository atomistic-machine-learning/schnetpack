from typing import Callable, Dict

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.structure as structure
import schnetpack.nn as snn

__all__ = ["PaiNN"]


def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021 (to appear)

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        activation=F.silu,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
        """
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.radial_basis = radial_basis

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        self.share_filters = shared_filters

        if shared_filters:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        self.interatomic_context_net = replicate_module(
            lambda: nn.Sequential(
                snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
                snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
            ),
            self.n_interactions,
            shared_interactions,
        )

        self.intraatomic_context_net = replicate_module(
            lambda: nn.Sequential(
                snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
                snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
            ),
            self.n_interactions,
            shared_interactions,
        )

        self.mu_channel_mix = replicate_module(
            lambda: snn.Dense(n_atom_basis, 2 * n_atom_basis, activation=None),
            self.n_interactions,
            shared_interactions,
        )

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
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        n_atoms = atomic_numbers.shape[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (
            interatomic_context_net,
            mu_channel_mix,
            intraatomic_context_net,
        ) in enumerate(
            zip(
                self.interatomic_context_net,
                self.mu_channel_mix,
                self.intraatomic_context_net,
            )
        ):
            ## inter-atomic
            x = interatomic_context_net(q)
            xj = x[idx_j]
            muj = mu[idx_j]

            x = filter_list[i] * xj

            # TochScript does not like this split in grad:
            # dq, dmuR, dmumu = torch.split(x, 128, dim=-1)
            dq = x[..., : self.n_atom_basis]
            dmuR = x[..., self.n_atom_basis : 2 * self.n_atom_basis]
            dmumu = x[..., 2 * self.n_atom_basis :]

            dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)
            dmu = dmuR * dir_ij[..., None] + dmumu * muj
            dmu = snn.scatter_add(dmu, idx_i, dim_size=n_atoms)

            q = q + dq
            mu = mu + dmu

            ## intra-atomic
            mu_mix = mu_channel_mix(mu)
            mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
            mu_Vn = torch.norm(mu_V, dim=-2, keepdim=True)

            ctx = torch.cat([q, mu_Vn], dim=-1)
            x = intraatomic_context_net(ctx)

            dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
            dmu_intra = dmu_intra * mu_W

            dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

            q = q + dq_intra + dqmu_intra
            mu = mu + dmu_intra

        q = q.squeeze(1)

        return {"scalar_representation": q, "vector_representation": mu}

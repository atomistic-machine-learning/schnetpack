import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
from torch.nn.init import zeros_, orthogonal_

from schnetpack.representation import AtomisticRepresentation


__all__ = ["PhysNetInteraction", "PhysNet"]


class PhysNetInteraction(nn.Module):
    def __init__(
        self,
        n_atom_basis,
        n_basis_functions=25,
        n_residuals_in=1,
        n_residuals_i=1,
        n_residuals_j=1,
        n_residuals_v=1,
        n_residuals_out=1,
        activation=spk.nn.Swish,
        cutoff=5.0,
        cutoff_network=spk.nn.MollifierCutoff,
    ):
        super(PhysNetInteraction, self).__init__()

        # attributes
        self.n_atom_basis = n_atom_basis

        # cutoff network
        cutoff_network = cutoff_network(cutoff)

        # input residual stack
        self.input_residual = spk.nn.ResidualStack(
            n_residuals_in, n_atom_basis, activation=activation
        )

        # i and j branches
        self.branch_i = nn.Sequential(
            spk.nn.ResidualStack(n_residuals_i, n_atom_basis, activation=activation),
            spk.nn.Dense(
                n_atom_basis,
                n_atom_basis,
                pre_activation=activation,
                weight_init=orthogonal_,
                bias_init=zeros_,
            ),
        )
        self.branch_j = nn.Sequential(
            spk.nn.ResidualStack(n_residuals_j, n_atom_basis, activation=activation),
            spk.nn.Dense(
                n_atom_basis,
                n_atom_basis,
                pre_activation=activation,
                weight_init=orthogonal_,
                bias_init=zeros_,
            ),
        )

        # convolution layer
        self.convolution_layer = spk.nn.BaseConvolutionLayer(
            filter_network=spk.nn.Dense(
                n_basis_functions, n_atom_basis, bias=False, weight_init=zeros_,
            ),
            cutoff_network=cutoff_network,
        )

        # merged v branch
        self.branch_v = nn.Sequential(
            spk.nn.ResidualStack(n_residuals_v, n_atom_basis, activation=activation),
            spk.nn.Dense(
                n_atom_basis,
                n_atom_basis,
                pre_activation=activation,
                weight_init=orthogonal_,
                bias_init=zeros_,
            ),
        )

        # charge and spin embeddings
        self.charge_embedding = nn.Parameter(torch.Tensor(n_atom_basis))
        self.spin_embedding = nn.Parameter(torch.Tensor(n_atom_basis))
        self.charge_keys = nn.Parameter(torch.Tensor(n_atom_basis))
        self.spin_keys = nn.Parameter(torch.Tensor(n_atom_basis))

        # output residual stack
        self.output_residual = spk.nn.ResidualStack(
            n_residuals_out, n_atom_basis, activation=activation
        )

        # reset parameters
        self.reset_parameters()

    def _attention_weights(self, x, key, charges, atom_mask):
        # compute weights
        w = F.softplus(torch.sum(x * (charges.sign() * key).unsqueeze(1), -1))
        w = w * atom_mask
        # compute weight norms
        wsum = w.sum(-1, keepdim=True)

        # return normalized weights; 1e-8 prevents possible division by 0
        return w / (wsum + 1e-8)

    def reset_parameters(self):
        nn.init.zeros_(self.charge_embedding)
        nn.init.zeros_(self.spin_embedding)
        nn.init.zeros_(self.charge_keys)
        nn.init.zeros_(self.spin_keys)

    def forward(
        self,
        x,
        r_ij,
        neighbors,
        neighbor_mask,
        f_ij=None,
        charges=None,
        spins=None,
        atom_mask=None,
        **kwargs,
    ):
        # todo: docstring
        # input residual stack
        x = self.input_residual(x)

        # i and j branches
        x_i = self.branch_i(x)
        x_j = self.branch_j(x)

        # cf-convolution of x_j and distances
        x_j = self.convolution_layer(
            y=x_j,
            r_ij=r_ij,
            neighbors=neighbors,
            pairwise_mask=neighbor_mask,
            f_ij=f_ij,
        )

        # merge x_i and x_j to v-branch
        v = x_i + x_j
        v = self.branch_v(v)

        # residual sum
        x = x + v

        # todo: check if attention layer can be used / cmul-layer can be used
        sfeatures, qfeatures = 0., 0.
        if charges is not None:
            charge_weights = self._attention_weights(x, self.charge_keys, charges,
                                                     atom_mask)
            qfeatures = (charges * charge_weights).unsqueeze(-1) * self.charge_embedding
        if spins is not None:
            spin_weights = self._attention_weights(x, self.spin_keys, spins, atom_mask)
            sfeatures = (spins * spin_weights).unsqueeze(-1) * self.spin_embedding

        x = x + sfeatures + qfeatures

        # output residual stack
        x = self.output_residual(x)

        return x


class PhysNet(AtomisticRepresentation):
    def __init__(
        self,
        n_atom_basis=128,
        n_basis_functions=32,
        n_interactions=6,
        n_residual_pre_x=1,
        n_residual_post_x=1,
        n_residual_pre_vi=1,
        n_residual_pre_vj=1,
        n_residual_post_v=1,
        n_residual_post_interaction=1,
        distance_expansion=None,
        cutoff=7.937658158457616,  # 15 Bohr converted to Angstrom
        activation=spk.nn.Swish,
        max_z=87,
        coupled_interactions=False,
        return_intermediate=True,
        return_distances=True,
        interaction_aggregation="sum",
        cutoff_network=spk.nn.MollifierCutoff,
        # todo: check if mollifier cutoff is available...
        # todo: basis func bernstein
        # todo: refactor n_gaussians to n_basis_funcs or similar
    ):

        # element specific bias for outputs
        embedding = spk.nn.Embedding(n_atom_basis, max_z)

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            distance_expansion = spk.nn.BernsteinPolynomials(n_basis_functions, cutoff,)

        # interaction blocks
        if coupled_interactions:
            interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_basis_functions=n_basis_functions,
                        n_residuals_in=n_residual_pre_x,
                        n_residuals_i=n_residual_pre_vi,
                        n_residuals_j=n_residual_pre_vj,
                        n_residuals_v=n_residual_post_v,
                        n_residuals_out=n_residual_post_x,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_basis_functions=n_basis_functions,
                        n_residuals_in=n_residual_pre_x,
                        n_residuals_i=n_residual_pre_vi,
                        n_residuals_j=n_residual_pre_vj,
                        n_residuals_v=n_residual_post_v,
                        n_residuals_out=n_residual_post_x,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                    )
                    for _ in range(n_interactions)
                ]
            )

        post_interactions = nn.ModuleList(
            [
                nn.Sequential(
                    spk.nn.ResidualStack(
                        n_features=n_atom_basis,
                        n_blocks=n_residual_post_interaction,
                        activation=activation,
                    ),
                    spk.nn.Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        pre_activation=activation,
                    ),
                )
                for _ in range(n_interactions)
            ]
        )

        # intermediate interactions to representation
        interaction_aggregation = spk.representation.InteractionAggregation(
            mode=interaction_aggregation
        )

        super(PhysNet, self).__init__(
            embedding=embedding,
            distance_expansion=distance_expansion,
            interactions=interactions,
            post_interactions=post_interactions,
            interaction_aggregation=interaction_aggregation,
            return_intermediate=return_intermediate,
            return_distances=return_distances,
            sum_before_interaction_append=False,
        )

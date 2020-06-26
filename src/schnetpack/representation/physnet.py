import torch
import torch.nn as nn
import schnetpack as spk
from torch.nn.init import zeros_, orthogonal_

from schnetpack.nn import AtomDistances
from schnetpack import Properties
from schnetpack.representation import AtomisticRepresentation


__all__ = ["PhysNetInteraction", "PhysNet"]


class PhysNetInteraction(nn.Module):
    def __init__(
        self,
        n_features,
        n_gaussians=25,
        n_residuals_in=1,
        n_residuals_i=1,
        n_residuals_j=1,
        n_residuals_v=1,
        n_residuals_out=1,
        activation=spk.nn.Swish,
        cutoff=None,
    ):
        super(PhysNetInteraction, self).__init__()

        # attributes
        self.n_atom_basis = n_features

        # input residual stack
        self.input_residual = spk.nn.ResidualStack(
            n_residuals_in, n_features, activation=activation
        )

        # i and j branches
        self.branch_i = nn.Sequential(
            spk.nn.ResidualStack(n_residuals_i, n_features, activation=activation),
            spk.nn.Dense(
                n_features,
                n_features,
                pre_activation=activation,
                weight_init=orthogonal_,
                bias_init=zeros_,
            ),
        )
        self.branch_j = nn.Sequential(
            spk.nn.ResidualStack(n_residuals_j, n_features, activation=activation),
            spk.nn.Dense(
                n_features,
                n_features,
                pre_activation=activation,
                weight_init=orthogonal_,
                bias_init=zeros_,
            ),
        )

        # convolution layer
        self.convolution_layer = spk.nn.BaseConvolutionLayer(
            filter_network=spk.nn.Dense(
                n_gaussians, n_features, bias=False, weight_init=zeros_
            ),
            # todo: add cutoff!
        )

        # merged v branch
        self.branch_v = nn.Sequential(
            spk.nn.ResidualStack(n_residuals_v, n_features, activation=activation),
            spk.nn.Dense(
                n_features,
                n_features,
                pre_activation=activation,
                weight_init=orthogonal_,
                bias_init=zeros_,
            ),
        )

        # output residual stack
        self.output_residual = spk.nn.ResidualStack(
            n_residuals_out, n_features, activation=activation
        )

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        # todo: use filter dimension like in spk?
        # todo: missing qfeatures and sfeatures from modular block
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

        # megre x_i and x_j to v-branch
        v = x_i + x_j
        v = self.branch_v(v)

        # residual sum
        x = x + v

        # output residual stack
        x = self.output_residual(x)

        return x


class PhysNet(AtomisticRepresentation):

    def __init__(
        self,
        n_atom_basis=128,
        n_gaussians=32,
        n_interactions=6,
        n_residual_pre_x=1,
        n_residual_post_x=1,
        n_residual_pre_vi=1,
        n_residual_pre_vj=1,
        n_residual_post_v=1,
        n_residual_post_interaction=1,
        distance_expansion=None,
        exp_weighting=True,
        cutoff=7.937658158457616,  # 15 Bohr converted to Angstrom
        lr_cutoff=None,
        activation=spk.nn.Swish,
        module_keep_prob=1.0,
        load_from=None,
        max_z=87,
        coupled_interactions=False,
        return_intermediate=True,
        return_distances=True,
        interaction_aggregation="sum",
        trainable_gaussians=False,
    ):

        # element specific bias for outputs
        embedding = spk.nn.Embedding(n_atom_basis, max_z)

        # layer for expanding interatomic distances in a basis
        # todo: rewrite other distance expansion functions
        if distance_expansion is None:
            distance_expansion = spk.nn.GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )

        # interaction blocks
        if coupled_interactions:
            interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_features=n_atom_basis,
                        n_gaussians=n_gaussians,
                        n_residuals_in=n_residual_pre_x,
                        n_residuals_i=n_residual_pre_vi,
                        n_residuals_j=n_residual_pre_vj,
                        n_residuals_v=n_residual_post_v,
                        n_residuals_out=n_residual_post_x,
                        activation=activation,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_features=n_atom_basis,
                        n_gaussians=n_gaussians,
                        n_residuals_in=n_residual_pre_x,
                        n_residuals_i=n_residual_pre_vi,
                        n_residuals_j=n_residual_pre_vj,
                        n_residuals_v=n_residual_post_v,
                        n_residuals_out=n_residual_post_x,
                        activation=activation,
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
    
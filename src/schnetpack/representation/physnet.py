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
        n_residuals_i=1,
        n_residuals_j=1,
        n_residuals_v=1,
        activation=spk.nn.Swish,
        cutoff=5.0,
        cutoff_network=spk.nn.MollifierCutoff,
    ):
        super(PhysNetInteraction, self).__init__()

        # attributes
        self.n_atom_basis = n_atom_basis

        # cutoff network
        cutoff_network = cutoff_network(cutoff)

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

        return v


class PhysNet(AtomisticRepresentation):
    def __init__(
        self,
        n_atom_basis=128,
        n_basis_functions=32,
        n_interactions=6,
        n_residual_pre_interaction=1,
        n_residual_pre_vi=1,
        n_residual_pre_vj=1,
        n_residual_post_v=1,
        n_residual_post_interaction=1,
        n_residual_interaction_output=1,
        distance_expansion=None,
        cutoff=7.937658158457616,  # 15 Bohr converted to Angstrom
        activation=spk.nn.Swish,
        max_z=87,
        coupled_interactions=False,
        return_intermediate=True,
        return_distances=True,
        interaction_aggregation="sum",
        cutoff_network=spk.nn.MollifierCutoff,
        charge_refinement=None,
        magmom_refinement=None,
        # todo: check if mollifier cutoff is available...
        # todo: basis func bernstein
    ):

        # element specific bias for outputs
        embedding = spk.nn.Embedding(n_atom_basis, max_z)

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            distance_expansion = spk.nn.BernsteinPolynomials(n_basis_functions, cutoff,)

        # pre interaction blocks
        if coupled_interactions:
            pre_interactions = nn.ModuleList(
                [
                    spk.nn.ResidualStack(
                        n_residual_pre_interaction, n_atom_basis, activation=activation
                    )
                ]
                * n_interactions
            )
        else:
            pre_interactions = nn.ModuleList(
                [
                    spk.nn.ResidualStack(
                        n_residual_pre_interaction, n_atom_basis, activation=activation
                    )
                    for _ in range(n_interactions)
                ]
            )

        # interaction blocks
        if coupled_interactions:
            interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_basis_functions=n_basis_functions,
                        n_residuals_i=n_residual_pre_vi,
                        n_residuals_j=n_residual_pre_vj,
                        n_residuals_v=n_residual_post_v,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                    )
                ]
                * n_interactions
            )
        else:
            interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_basis_functions=n_basis_functions,
                        n_residuals_i=n_residual_pre_vi,
                        n_residuals_j=n_residual_pre_vj,
                        n_residuals_v=n_residual_post_v,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # refinement blocks
        if coupled_interactions:
            refinement_layers = []
            if charge_refinement is not None:
                refinement_layers.append(
                    spk.representation.InteractionRefinement(
                        n_features=n_atom_basis, property=charge_refinement
                    )
                )
            if magmom_refinement is not None:
                refinement_layers.append(
                    spk.representation.InteractionRefinement(
                        n_features=n_atom_basis, property=magmom_refinement
                    )
                )
            refinement_block = spk.nn.FeatureSum(layers=refinement_layers)
            refinement_blocks = nn.ModuleList([refinement_block] * n_interactions)
        else:
            refinement_blocks = []
            for _ in range(n_interactions):
                refinement_layers = []
                if charge_refinement is not None:
                    refinement_layers.append(
                        spk.representation.InteractionRefinement(
                            n_features=n_atom_basis, property=charge_refinement
                        )
                    )
                if magmom_refinement is not None:
                    refinement_layers.append(
                        spk.representation.InteractionRefinement(
                            n_features=n_atom_basis, property=magmom_refinement
                        )
                    )
                refinement_block = spk.nn.FeatureSum(layers=refinement_layers)
                refinement_blocks.append(refinement_block)
            refinement_blocks = nn.ModuleList(refinement_blocks)

        # post-interaction block
        if coupled_interactions:
            post_interactions = nn.ModuleList(
                [
                    spk.nn.ResidualStack(
                        n_residual_post_interaction, n_atom_basis, activation=activation
                    )
                ]
                * n_interactions
            )
        else:
            post_interactions = nn.ModuleList(
                [
                    spk.nn.ResidualStack(
                        n_residual_post_interaction, n_atom_basis, activation=activation
                    )
                    for _ in range(n_interactions)
                ]
            )

        # interaction output filter
        if coupled_interactions:
            interaction_outputs = nn.ModuleList(
                [
                    nn.Sequential(
                        spk.nn.ResidualStack(
                            n_features=n_atom_basis,
                            n_blocks=n_residual_interaction_output,
                            activation=activation,
                        ),
                        spk.nn.Dense(
                            in_features=n_atom_basis,
                            out_features=n_atom_basis,
                            pre_activation=activation,
                        ),
                    )
                ]
                * n_interactions
            )
        else:
            interaction_outputs = nn.ModuleList(
                [
                    nn.Sequential(
                        spk.nn.ResidualStack(
                            n_features=n_atom_basis,
                            n_blocks=n_residual_interaction_output,
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
            pre_interactions=pre_interactions,
            interaction_refinements=refinement_blocks,
            post_interactions=post_interactions,
            interaction_outputs=interaction_outputs,
            interaction_aggregation=interaction_aggregation,
            return_intermediate=return_intermediate,
            return_distances=return_distances,
            sum_before_interaction_append=False,
        )

import torch
import torch.nn as nn
import schnetpack as spk
from torch.nn.init import zeros_, orthogonal_

from schnetpack.nn import AtomDistances
from schnetpack import Properties


__all__ = ["PhysNetInteraction", "PhysNet"]


class PhysNetInteraction(nn.Module):
    def __init__(
        self,
        n_features,
        n_basis_functions,
        n_residuals_in,
        n_residuals_i,
        n_residuals_j,
        n_residuals_v,
        n_residuals_out,
        activation=spk.nn.Swish,
    ):
        super(PhysNetInteraction, self).__init__()

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
                n_basis_functions, n_features, bias=False, weight_init=zeros_
            ),
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


class PhysNet(nn.Module):
    def __init__(
        self,
        n_features=128,
        n_basis_functions=32,
        n_interactions=6,
        n_residual_pre_x=1,
        n_residual_post_x=1,
        n_residual_pre_vi=1,
        n_residual_pre_vj=1,
        n_residual_post_v=1,
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
    ):
        self.register_parameter("element_bias", nn.Parameter(torch.Tensor(max_z)))
        # element specific bias for outputs
        self.embedding = spk.nn.Embedding(n_features, max_z)

        # layer for computing interatomic distances
        self.distances = AtomDistances()

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            distance_expansion = spk.nn.ExponentialBernsteinPolynomials(
                self.num_basis_functions, exp_weighting=self.exp_weighting
            )
        self.distance_expansion = distance_expansion

        # interaction blocks
        if coupled_interactions:
            self.interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_features=n_features,
                        n_basis_functions=n_basis_functions,
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
            self.interactions = nn.ModuleList(
                [
                    PhysNetInteraction(
                        n_features=n_features,
                        n_basis_functions=n_basis_functions,
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

        # set attributes
        self.return_intermediate = return_intermediate
        self.return_distances = return_distances

    def forward(self, inputs):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            dict: atomic features representations; intermediate features if
            return_intermediate==True; distance matrix if return_distances==True

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        # compute interatomic distance and distance expansion
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        f_ij = self.distance_expansion(r_ij)

        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
            if self.return_intermediate:
                # todo: eventually append(x)? or append(v) in schnet?
                xs.append(v)

        # build results dict
        results = dict(representation=x)
        if self.return_intermediate:
            results.update(dict(intermediate_interactions=xs))
        if self.return_distances:
            results.update(dict(distances=r_ij))

        return results

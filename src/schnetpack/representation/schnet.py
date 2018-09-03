import torch.nn as nn

import schnetpack.nn.acsf
import schnetpack.nn.activations
import schnetpack.nn.base
import schnetpack.nn.cfconv
import schnetpack.nn.neighbors
from schnetpack.data import Structure


class SchNetInteraction(nn.Module):
    """
    SchNet interaction block for modeling quantum interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_basis, n_spatial_basis, n_filters,
                 normalize_filter=False):
        super(SchNetInteraction, self).__init__()

        # initialize filters
        self.filter_network = nn.Sequential(
            schnetpack.nn.base.Dense(n_spatial_basis, n_filters,
                                     activation=schnetpack.nn.activations.shifted_softplus),
            schnetpack.nn.base.Dense(n_filters, n_filters)
        )

        # initialize interaction blocks
        self.cfconv = schnetpack.nn.cfconv.CFConv(n_atom_basis, n_filters, n_atom_basis,
                                                  self.filter_network,
                                                  activation=schnetpack.nn.activations.shifted_softplus,
                                                  normalize_filter=normalize_filter)
        self.dense = schnetpack.nn.base.Dense(n_atom_basis, n_atom_basis)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Atom-wise input representations.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            neighbor_mask (torch.Tensor): Mask to indicate virtual neighbors introduced via zeros padding.
            f_ij (torch.Tensor): Use at your own risk.

        Returns:
            torch.Tensor: SchNet representation.
        """
        v = self.cfconv.forward(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
        v = self.dense.forward(v)
        return v


class SchNetCutoffInteraction(nn.Module):
    """
    SchNet interaction block for modeling quantum interactions of atomistic systems with cosine cutoff.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_basis, n_spatial_basis, n_filters, cutoff,
                 normalize_filter=False):
        super(SchNetCutoffInteraction, self).__init__()

        # initialize filters
        self.filter_network = nn.Sequential(
            schnetpack.nn.base.Dense(n_spatial_basis, n_filters,
                                     activation=schnetpack.nn.activations.shifted_softplus),
            schnetpack.nn.base.Dense(n_filters, n_filters)
        )

        self.cutoff_network = schnetpack.nn.CosineCutoff(cutoff)

        # initialize interaction blocks
        self.cfconv = schnetpack.nn.cfconv.CFConv(n_atom_basis, n_filters, n_atom_basis,
                                                  self.filter_network,
                                                  cutoff_network=self.cutoff_network,
                                                  activation=schnetpack.nn.activations.shifted_softplus,
                                                  normalize_filter=normalize_filter)
        self.dense = schnetpack.nn.base.Dense(n_atom_basis, n_atom_basis)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Atom-wise input representations.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            neighbor_mask (torch.Tensor): Mask to indicate virtual neighbors introduced via zeros padding.
            f_ij (torch.Tensor): Use at your own risk.

        Returns:
            torch.Tensor: SchNet representation.
        """
        v = self.cfconv.forward(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense.forward(v)
        return v


class SchNet(nn.Module):
    """
    SchNet architecture for learning representations of atomistic systems
    as described in [#schnet1]_ [#schnet2]_ [#schnet3]_

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_filters (int): number of filters used in continuous-filter convolution
        n_interactions (int): number of interaction blocks
        cutoff (float): cutoff radius of filters
        n_gaussians (int): number of Gaussians which are used to expand atom distances
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
        coupled_interactions (bool): if true, share the weights across interaction blocks
            and filter-generating networks.
        return_intermediate (bool): if true, also return intermediate feature representations
            after each interaction block
        max_z (int): maximum allowed nuclear charge in dataset. This determines the size of the embedding matrix.

    References
    ----------
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet2] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    """

    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=1, cutoff=5.0, n_gaussians=25,
                 normalize_filter=False, coupled_interactions=False,
                 return_intermediate=False, max_z=100, interaction_block=SchNetInteraction, trainable_gaussians=False,
                 distance_expansion=None):
        super(SchNet, self).__init__()

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # spatial features
        self.distances = schnetpack.nn.neighbors.AtomDistances()
        if distance_expansion is None:
            self.distance_expansion = schnetpack.nn.acsf.GaussianSmearing(0.0, cutoff, n_gaussians,
                                                                          trainable=trainable_gaussians)
        else:
            self.distance_expansion = distance_expansion

        self.return_intermediate = return_intermediate

        # interaction network
        if coupled_interactions:
            self.interactions = nn.ModuleList([
                                                  interaction_block(n_atom_basis=n_atom_basis,
                                                                    n_spatial_basis=n_gaussians,
                                                                    n_filters=n_filters,
                                                                    normalize_filter=normalize_filter)
                                              ] * n_interactions)
        else:
            self.interactions = nn.ModuleList([
                interaction_block(n_atom_basis=n_atom_basis, n_spatial_basis=n_gaussians,
                                  n_filters=n_filters, normalize_filter=normalize_filter)
                for _ in range(n_interactions)
            ])

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Final Atom-wise SchNet representation.
            torch.Tensor: Atom-wise SchNet representation of intermediate layers.
        """
        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        cell = inputs[Structure.cell]
        cell_offset = inputs[Structure.cell_offset]
        neighbors = inputs[Structure.neighbors]
        neighbor_mask = inputs[Structure.neighbor_mask]

        # atom embedding
        x = self.embedding(atomic_numbers)

        # spatial features
        r_ij = self.distances(positions, neighbors, cell, cell_offset)
        f_ij = self.distance_expansion(r_ij)

        # interactions
        if self.return_intermediate:
            xs = [x]

        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v

            if self.return_intermediate:
                xs.append(xs)

        if self.return_intermediate:
            return x, xs
        return x

import torch
import torch.nn as nn

from schnetpack.nn.base import Dense
from schnetpack.data import Structure
from schnetpack.nn.cfconv import CFConv
from schnetpack.nn.cutoff import HardCutoff
from schnetpack.nn.acsf import GaussianSmearing
from schnetpack.nn.neighbors import AtomDistances
from schnetpack.nn.activations import shifted_softplus


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff,
        cutoff_network=HardCutoff,
        normalize_filter=False,
    ):
        super(SchNetInteraction, self).__init__()
        # filter block used in interaction block
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = CFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v


class SchNet(nn.Module):
    """
    SchNet architecture for learning representations of atomistic systems
    as described in [#schnet1]_ [#schnet_transfer]_ [#schnet3]_

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_filters (int): number of filters used in continuous-filter convolution
        n_interactions (int): number of interaction blocks
        cutoff (float): cutoff radius of filters
        n_gaussians (int): number of Gaussians which are used to expand atom distances
        normalize_filter (bool): if true, divide filter by number of neighbors
            over which convolution is applied
        coupled_interactions (bool): if true, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool): if true, also return intermediate feature
            representations after each interaction block
        max_z (int): maximum allowed nuclear charge in dataset. This determines
            the size of the embedding matrix.

    References
    ----------
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=1,
        cutoff=5.0,
        n_gaussians=25,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        cutoff_network=HardCutoff,
        trainable_gaussians=False,
        distance_expansion=None,
        charged_systems=False,
    ):
        super(SchNet, self).__init__()

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # spatial features
        self.distances = AtomDistances()
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion

        self.return_intermediate = return_intermediate

        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

        # interaction network
        if coupled_interactions:
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

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
        atom_mask = inputs[Structure.atom_mask]

        # atom embedding
        x = self.embedding(atomic_numbers)

        if False and self.charged_systems and Structure.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Structure.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # spatial features
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        f_ij = self.distance_expansion(r_ij)

        # interactions
        if self.return_intermediate:
            xs = [x]

        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v

            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x

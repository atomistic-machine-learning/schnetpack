from typing import Callable, Dict

import torch
from torch import nn

import schnetpack as spk
import schnetpack.structure as structure
from schnetpack.nn import Dense, CFConv
from schnetpack.nn.activations import shifted_softplus


class SchNetInteraction(nn.Module):
    r""" SchNet interaction block for modeling interactions of atomistic systems. """

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        normalize_filter: bool = False,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            normalize_filter: if True, divide aggregated filter by number
                of neighbors over which convolution is applied.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.cfconv = CFConv(reduce="mean" if normalize_filter else "sum")
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation),
            Dense(n_filters, n_filters),
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]
        x = self.cfconv(x, Wij, idx_i, idx_j)
        x = self.f2out(x)
        return x


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems.

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int = None,
        normalize_filter: bool = False,
        coupled_interactions: bool = False,
        return_intermediate: bool = False,
        max_z: int = 100,
        charged_systems: bool = False,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            normalize_filter: if True, divide aggregated filter by number
                of neighbors over which convolution is applied.
            coupled_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            return_intermediate:
            max_z:
            charged_systems:
            activation:
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn

        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, self.n_atom_basis))
            self.charge.data.normal_(0, 1.0 / self.n_atom_basis ** 0.5)

        # layers
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=self.n_atom_basis,
                        n_rbf=self.radial_basis.n_rbf,
                        n_filters=self.n_filters,
                        normalize_filter=normalize_filter,
                        activation=activation,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=self.n_atom_basis,
                        n_rbf=self.radial_basis.n_rbf,
                        n_filters=self.n_filters,
                        normalize_filter=normalize_filter,
                        activation=activation,
                    )
                    for _ in range(n_interactions)
                ]
            )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]

        # compute atom and pair features
        x = self.embedding(atomic_numbers)
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        return {"scalar_representation": x}

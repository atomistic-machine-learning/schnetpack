from typing import Callable, Dict, Union

import torch
from torch import nn

import schnetpack.properties as structure
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.embedding import NuclearEmbedding
from schnetpack.nn import ElectronicEmbedding
from schnetpack.nn.activations import shifted_softplus

import schnetpack.nn as snn

__all__ = ["SchNet", "SchNetInteraction"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
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

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x



class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems

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
        shared_interactions: bool = False,
        max_z: int = 100,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        activate_charge_spin_embedding: bool = False,
        nuclear_embedding: Union[Callable, nn.Module] = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
            activate_charge_spin_embedding: if True, charge and spin embeddings are added to nuclear embeddings taken from SpookyNet Implementation
            nuclear_embedding: type of nuclear embedding to use (simple is simple embedding and complex is the one with electron configuration)
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.activate_charge_spin_embedding = activate_charge_spin_embedding
        self.nuclear_embedding = nuclear_embedding
        # nuclear embedding layer (often complex nuclear embedding has negative impact on the performance of the model, so we can use simple nuclear embedding to avoid this issue)
        if self.nuclear_embedding is None:
            self.nuclear_embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        if self.activate_charge_spin_embedding:
        # needed for spin and charge embeeding
            # additional embeedings for the total charge
            self.charge_embedding = ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=True)
            # additional embeedings for the spin multiplicity
            self.magmom_embedding = ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=False)

        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):

        # get inputs
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        num_batch = len(inputs[structure.idx])
        # batch seq
        # Index for each atom that specifies to which molecule in the
        # batch it belongs. For example, when predicting a H2O and a CH4
        # molecule, batch_seg would be [0, 0, 0, 1, 1, 1, 1, 1] to
        # indicate that the first three atoms belong to the first molecule
        # and the last five atoms to the second molecule.
        batch_seg = inputs[structure.idx_m]

        # compute pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute atom and pair features
        x = self.nuclear_embedding(atomic_numbers)
        if self.activate_charge_spin_embedding:
            # get inputs for spin and charge embedding 
            # (to avoid error not having total charge /spin multiplicity in db if embedding not used)
            total_charge = inputs[structure.total_charge]
            spin = inputs[structure.spin_multiplicity]

            # specific total charge embeeding
            charge_embedding = self.charge_embedding(x,total_charge,num_batch,batch_seg)

            # specific spin embeeding
            spin_embedding = self.magmom_embedding(x,spin,num_batch,batch_seg)

            # additive combining nuclear, charge and spin embeeding
            x = x + charge_embedding + spin_embedding


        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        inputs["scalar_representation"] = x
        return inputs

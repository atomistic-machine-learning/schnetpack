"""
conditional_schnet.py

Conditional SchNet that conditions on dataset_id via additive embedding
at the input level. Everything else — interactions, output, forces via
autograd — is identical to the standard SchNet.

The only change vs standard SchNet:
    x = embedding(Z) + dataset_embedding(dataset_id_per_atom)

This lets the model learn dataset-specific atomic representations while
sharing all interaction blocks and output layers across datasets.

Usage
-----
    model = ConditionalSchNet(
        n_atom_basis=64,
        n_interactions=3,
        n_datasets=2,          # number of component datasets in MergedDataset
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )
"""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import schnetpack.properties as properties
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
import schnetpack.nn as snn

__all__ = ["ConditionalSchNet", "ConditionalSchNetInteraction"]


class ConditionalSchNetInteraction(nn.Module):
    """
    Standard SchNet interaction block — unchanged from original.
    Kept as a separate class for clarity.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        super().__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
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
    ) -> torch.Tensor:
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
        x = self.f2out(x)
        return x


class ConditionalSchNet(nn.Module):
    """
    Conditional SchNet.

    Identical to standard SchNet except the initial atomic embedding
    is augmented with a dataset-specific embedding:

        x = nuclear_embedding(Z) + dataset_embedding(dataset_id)

    dataset_embedding has shape [n_datasets, n_atom_basis] — same
    dimension as the nuclear embedding so addition works directly.
    dataset_id is per-molecule (shape [n_molecules]) and is expanded
    to per-atom using idx_m from the collated batch.

    All interaction blocks, output layers, and force computation via
    autograd are identical to standard SchNet.

    Args:
        n_atom_basis: size of atomic embedding vectors.
        n_interactions: number of interaction blocks.
        n_datasets: number of component datasets (size of dataset_id vocabulary).
                    Must match the number of datasets in MergedDataset.
        radial_basis: layer for expanding interatomic distances.
        cutoff_fn: cutoff function.
        n_filters: number of filters in continuous-filter convolution.
                   Defaults to n_atom_basis.
        shared_interactions: share weights across interaction blocks.
        activation: activation function.
        nuclear_embedding: custom nuclear embedding. Defaults to
                           nn.Embedding(100, n_atom_basis).
        electronic_embeddings: list of additional electronic embeddings.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_datasets: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int = None,
        shared_interactions: bool = False,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
    ):
        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff

        # Nuclear embedding — same as standard SchNet
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding

        # Dataset-conditional embedding — maps dataset_id → n_atom_basis vector
        # Added to nuclear embedding before interactions
        self.dataset_embedding = nn.Embedding(n_datasets, n_atom_basis)

        # Initialize dataset embedding to zero so at the start of training
        # the model behaves like a standard SchNet — conditioning is learned
        # gradually from scratch
        nn.init.zeros_(self.dataset_embedding.weight)

        # Electronic embeddings — same as standard SchNet
        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings)

        # Interaction blocks — identical to standard SchNet
        self.interactions = snn.replicate_module(
            lambda: ConditionalSchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Standard inputs
        #print(f"forward dataset_id: {inputs['dataset_id'].squeeze().tolist()}")
        atomic_numbers = inputs[properties.Z]       # [n_atoms]
        r_ij = inputs[properties.Rij]               # [n_pairs, 3]
        idx_i = inputs[properties.idx_i]            # [n_pairs]
        idx_j = inputs[properties.idx_j]            # [n_pairs]
        idx_m = inputs[properties.idx_m]            # [n_atoms] molecule index per atom

        # dataset_id is per-molecule: shape [n_molecules, 1] or [n_molecules]
        dataset_id = inputs["dataset_id"].squeeze(-1)  # [n_molecules]
        
        # Expand dataset_id from per-molecule to per-atom using idx_m
        # idx_m[i] = molecule index of atom i
        dataset_id_per_atom = dataset_id[idx_m]    # [n_atoms]


        # Compute pair features — same as standard SchNet
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # Initial atomic embedding
        x = self.embedding(atomic_numbers)          # [n_atoms, n_atom_basis]

        # Add dataset-conditional embedding — the only change vs standard SchNet
        x = x + self.dataset_embedding(dataset_id_per_atom)  # [n_atoms, n_atom_basis]

        # Electronic embeddings — same as standard SchNet
        for embedding in self.electronic_embeddings:
            x = x + embedding(x, inputs)

        # Interaction blocks — same as standard SchNet
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        # Store scalar representation — same key as standard SchNet
        # so output modules (Atomwise, Forces) work without any changes
        inputs["scalar_representation"] = x

        return inputs
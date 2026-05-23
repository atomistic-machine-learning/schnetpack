from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import schnetpack.properties as properties
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
import schnetpack.nn as snn

__all__ = ["ConditionalSchNet", "ConditionalSchNetInteraction"]

CONDITIONING_MODES = ("input", "mlp_layer")


class ConditionalSchNetInteraction(nn.Module):
    """
    Standard SchNet interaction block — identical to original.
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


class ConditioningMLP(nn.Module):
    """
    Small MLP that maps a raw dataset embedding to a conditioning vector y.

    Architecture:
        dataset_emb (n_atom_basis)
            → Linear(n_atom_basis, n_atom_basis)
            → shifted_softplus
            → Linear(n_atom_basis, n_atom_basis)
            → y  (n_atom_basis)

    Used in "mlp_layer" mode only.

    NOTE: We do NOT zero-init the MLP weights here. The dataset_embedding
    itself is zero-initialized in ConditionalSchNet, so at the start of
    training the MLP receives a zero input and produces a fixed learned
    bias — but crucially, gradients still flow back through both the MLP
    and the embedding, allowing both to update from step 1.

    Zero-initializing the MLP's last layer on top of a zero-init embedding
    causes a dead-gradient trap where nothing ever learns.
    """

    def __init__(self, n_atom_basis: int, activation: Callable = shifted_softplus):
        super().__init__()
        self.net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        # Use default Kaiming uniform init (PyTorch default for Linear/Dense).
        # Do NOT zero-init here — see class docstring.

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class ConditionalSchNet(nn.Module):
    """
    Conditional SchNet with switchable conditioning mode.

    n_atom_basis : int
        Size of atomic embedding vectors.
    n_interactions : int
        Number of interaction blocks.
    n_datasets : int
        Number of component datasets — size of the dataset_id vocabulary.
        Must match the number of datasets in MergedDataset.
    radial_basis : nn.Module
        Layer for expanding interatomic distances in a basis set.
    cutoff_fn : Callable
        Cutoff function.
    conditioning_mode : str
        "input"     — add dataset embedding once at input (default).
        "mlp_layer" — pass embedding through MLP, inject at input and
                      after every interaction layer.
    n_filters : int, optional
        Number of filters in cfconv. Defaults to n_atom_basis.
    shared_interactions : bool
        Share weights across interaction blocks.
    activation : Callable
        Activation function.
    nuclear_embedding : nn.Module, optional
        Custom nuclear embedding. Defaults to nn.Embedding(100, n_atom_basis).
    electronic_embeddings : list, optional
        Additional electronic embeddings (e.g. spin, charge).
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_datasets: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        conditioning_mode: str = "input",
        n_filters: int = None,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
    ):
        super().__init__()

        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode must be one of {CONDITIONING_MODES}, "
                f"got '{conditioning_mode}'."
            )

        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.conditioning_mode = conditioning_mode

        # --- Nuclear embedding (same as standard SchNet) ---
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding

        # --- Dataset embedding ---
        # Shape: [n_datasets, n_atom_basis]
        # Zero-initialized so model starts as standard SchNet at epoch 0.
        # In "input" mode:     y = dataset_embedding[id]        (direct use)
        # In "mlp_layer" mode: y = MLP(dataset_embedding[id])   (MLP uses default init,
        #                      so gradients flow immediately from step 1)
        self.dataset_embedding = nn.Embedding(n_datasets, n_atom_basis)
        nn.init.zeros_(self.dataset_embedding.weight)

        # --- Conditioning MLP (mlp_layer mode only) ---
        # MLP uses DEFAULT (Kaiming) init — not zero-init.
        # See ConditioningMLP docstring for explanation.
        if conditioning_mode == "mlp_layer":
            self.conditioning_mlp = ConditioningMLP(n_atom_basis, activation)
        else:
            self.conditioning_mlp = None

        # --- Electronic embeddings (same as standard SchNet) ---
        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings)

        # --- Interaction blocks (same as standard SchNet) ---
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

    def _get_conditioning_vector(
        self, dataset_id_per_atom: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditioning vector y for each atom.

        "input" mode:
            y = dataset_embedding[dataset_id]        shape [n_atoms, n_atom_basis]

        "mlp_layer" mode:
            y = MLP(dataset_embedding[dataset_id])   shape [n_atoms, n_atom_basis]
        """
        emb = self.dataset_embedding(dataset_id_per_atom)  # [n_atoms, n_atom_basis]
        if self.conditioning_mode == "mlp_layer":
            return self.conditioning_mlp(emb)
        return emb

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Standard inputs ---
        atomic_numbers = inputs[properties.Z]  # [n_atoms]
        r_ij = inputs[properties.Rij]  # [n_pairs, 3]
        idx_i = inputs[properties.idx_i]  # [n_pairs]
        idx_j = inputs[properties.idx_j]  # [n_pairs]
        idx_m = inputs[properties.idx_m]  # [n_atoms]

        # dataset_id: per-molecule → expand to per-atom via idx_m
        dataset_id = inputs["dataset_id"].squeeze(-1)  # [n_molecules]
        dataset_id_per_atom = dataset_id[idx_m]  # [n_atoms]

        # --- Pair features (same as standard SchNet) ---
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # --- Conditioning vector y ---
        # Computed once, reused at every layer in mlp_layer mode.
        # In "input" mode:     y = dataset_embedding[id]
        # In "mlp_layer" mode: y = MLP(dataset_embedding[id])
        y = self._get_conditioning_vector(
            dataset_id_per_atom
        )  # [n_atoms, n_atom_basis]

        # --- Initial atomic embedding ---
        x = self.embedding(atomic_numbers)  # [n_atoms, n_atom_basis]
        x = x + y  # inject conditioning at input (both modes)

        # --- Electronic embeddings (same as standard SchNet) ---
        for emb in self.electronic_embeddings:
            x = x + emb(x, inputs)

        # --- Interaction blocks ---
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)

            if self.conditioning_mode == "mlp_layer":
                # Re-inject conditioning after every interaction (supervisor's approach)
                x = x + v + y
            else:
                # Standard residual — no re-injection ("input" mode)
                x = x + v

        inputs["scalar_representation"] = x
        return inputs

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import schnetpack.properties as properties
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
import schnetpack.nn as snn

__all__ = ["ConditionalSchNet", "ConditionalSchNetInteraction"]

CONDITIONING_MODES = ("input", "mlp_layer", "multi_head")


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

    """

    def __init__(self, n_atom_basis: int, activation: Callable = shifted_softplus):
        super().__init__()
        self.net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class DatasetHead(nn.Module):
    """
    Per-dataset output projection applied after all interaction blocks.

    Architecture (n_head_layers=2):
        x [n_atom_basis]
            → Linear(n_atom_basis, n_atom_basis) → shifted_softplus
            → Linear(n_atom_basis, n_atom_basis)
            → x_projected [n_atom_basis]

    Residual connection: x_out = x + head(x)
    This ensures the head starts as an identity map (if last layer is
    zero-initialized) and learns a correction on top of the shared repr.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_layers: int = 2,
        activation: Callable = shifted_softplus,
    ):
        super().__init__()
        assert n_layers >= 1, "n_head_layers must be >= 1"

        layers = []
        for i in range(n_layers):
            act = activation if i < n_layers - 1 else None
            layers.append(Dense(n_atom_basis, n_atom_basis, activation=act))
        self.net = nn.Sequential(*layers)

        # Zero-init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual


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
        "multi_head" — add per-dataset embedding once at input,
                      apply per-dataset head after interaction blocks.
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
        n_head_layers: int = 2,
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

        self.dataset_embedding = nn.Embedding(n_datasets, n_atom_basis)

        # --- Conditioning MLP (mlp_layer mode only) ---
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
        self.dataset_heads = None

        if conditioning_mode == "multi_head":
            self.dataset_heads = nn.ModuleList(
                [
                    DatasetHead(
                        n_atom_basis, n_layers=n_head_layers, activation=activation
                    )
                    for _ in range(n_datasets)
                ]
            )

    def _get_conditioning_vector(
        self, dataset_id_per_atom: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Returns None for multi_head — no input injection needed.
        """
        if self.conditioning_mode == "multi_head":
            return None

        emb = self.dataset_embedding(dataset_id_per_atom)
        if self.conditioning_mode == "mlp_layer":
            return self.conditioning_mlp(emb)
        return emb

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        idx_m = inputs[properties.idx_m]

        dataset_id = inputs["dataset_id"].squeeze(-1)
        dataset_id_per_atom = dataset_id[idx_m]

        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        y = self._get_conditioning_vector(dataset_id_per_atom)  # None for multi_head

        x = self.embedding(atomic_numbers)

        # ── input injection
        if self.conditioning_mode == "input":
            x = x + y

        for emb in self.electronic_embeddings:
            x = x + emb(x, inputs)

        # ── interaction blocks
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        # ── post-interaction shift (mlp_layer mode only)
        if self.conditioning_mode == "mlp_layer":
            x = x + y

        # ── per-dataset head (multi_head only)
        if self.conditioning_mode == "multi_head":
            x_out = torch.empty_like(x)
            for d, head in enumerate(self.dataset_heads):
                mask = dataset_id_per_atom == d
                if mask.any():
                    x_out[mask] = head(x[mask])
            x = x_out

        inputs["scalar_representation"] = x
        return inputs

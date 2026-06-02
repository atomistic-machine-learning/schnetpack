from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
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
    Maps a dataset embedding to a conditioning vector y.
    Used in mlp_layer mode — output is added to scalar_representation
    after all interaction blocks.

    Architecture:
        emb [n_atom_basis]
            → Dense → shifted_softplus
            → Dense
            → y [n_atom_basis]
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
    Per-dataset readout MLP. Takes the shared atomic representation and
    predicts a per-atom energy scalar directly.

    Architecture:
        x [n_atom_basis]
            → Dense → silu
            → Dense
            → atomic energy [1]

    All heads are run on ALL atoms; the correct prediction is selected
    afterwards by dataset_id, avoiding any masking during the forward pass.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns [n_atoms, 1]
        return self.outnet(x)


class ConditionalSchNet(nn.Module):
    """
    Conditional SchNet with switchable conditioning mode.

    Modes
    -----
    input       : dataset embedding added once at atom embedding.
    mlp_layer   : backbone runs vanilla SchNet, dataset embedding passed
                  through ConditioningMLP and added to scalar_representation
                  after all interaction blocks.
    multi_head  : backbone runs vanilla SchNet, per-dataset DatasetHead MLPs
                  produce atomic energies directly. Use MultiHeadAtomwise as
                  the output module instead of Atomwise.

    Args:
        n_atom_basis: size of atomic embedding vectors.
        n_interactions: number of interaction blocks.
        n_datasets: number of component datasets.
        radial_basis: layer for expanding interatomic distances.
        cutoff_fn: cutoff function.
        conditioning_mode: one of ("input", "mlp_layer", "multi_head").
        n_filters: number of filters in cfconv. Defaults to n_atom_basis.
        shared_interactions: share weights across interaction blocks.
        shared_filters: share filter network weights.
        activation: activation function.
        nuclear_embedding: custom nuclear embedding.
        electronic_embeddings: additional electronic embeddings.
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
        self.n_datasets = n_datasets
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.conditioning_mode = conditioning_mode

        # --- Nuclear embedding ---
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding

        # --- Dataset embedding (input + mlp_layer modes only) ---
        self.dataset_embedding = None
        self.conditioning_mlp = None
        if conditioning_mode in ("input", "mlp_layer"):
            self.dataset_embedding = nn.Embedding(n_datasets, n_atom_basis)
            if conditioning_mode == "mlp_layer":
                self.conditioning_mlp = ConditioningMLP(n_atom_basis, activation)

        # --- Electronic embeddings ---
        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings)

        # --- Interaction blocks ---
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
    ) -> Optional[torch.Tensor]:
        """Compute y for input / mlp_layer modes. Returns None for multi_head."""
        if self.dataset_embedding is None:
            return None
        emb = self.dataset_embedding(dataset_id_per_atom)
        if self.conditioning_mlp is not None:
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

        # conditioning vector (None for multi_head)
        y = self._get_conditioning_vector(dataset_id_per_atom)

        # --- atom embedding ---
        x = self.embedding(atomic_numbers)

        # input mode: inject dataset signal at embedding
        if self.conditioning_mode == "input":
            x = x + y

        for emb in self.electronic_embeddings:
            x = x + emb(x, inputs)

        # --- interaction blocks (vanilla SchNet for all modes) ---
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        # mlp_layer mode: inject dataset signal at scalar_representation
        if self.conditioning_mode == "mlp_layer":
            x = x + y

        inputs["scalar_representation"] = x
        return inputs

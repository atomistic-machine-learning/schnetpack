from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.nn.activations import shifted_softplus
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing
from schnetpack.representation.cschnet import DatasetHead

__all__ = ["ConditionalPaiNN"]

CONDITIONING_MODES = ("input", "output_shift", "multi_head")


class ConditionalPaiNN(nn.Module):
    """
    Conditional PaiNN with switchable conditioning mode.

    Modes
    -----
    input       : dataset embedding added once to scalar embedding q at input.
    output_shift   : backbone runs vanilla PaiNN, dataset embedding added directly
                  to scalar_representation after all interaction+mixing blocks.
    multi_head  : backbone runs vanilla PaiNN, per-dataset DatasetHead MLPs
                  produce atomic energies directly. Use MultiHeadAtomwise as
                  the output module instead of Atomwise.

    Args:
        n_atom_basis: number of features to describe atomic environments.
        n_interactions: number of interaction blocks.
        n_datasets: number of component datasets.
        radial_basis: layer for expanding interatomic distances in a basis set.
        cutoff_fn: cutoff function.
        conditioning_mode: one of ("input", "output_shift", "multi_head").
        activation: activation function.
        shared_interactions: share weights across interaction blocks.
        shared_filters: share filter network weights.
        epsilon: numerical stability parameter for PaiNNMixing.
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
        activation: Optional[Callable] = F.silu,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
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
        self.n_interactions = n_interactions
        self.n_datasets = n_datasets
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.conditioning_mode = conditioning_mode

        # --- Nuclear embedding ---
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding

        # --- Dataset embedding (input + output_shift modes only) ---
        # input mode    : added to nuclear embedding before message passing.
        # output_shift mode: added directly to scalar_representation after all blocks.
        self.dataset_embedding = None
        if conditioning_mode in ("input", "output_shift"):
            self.dataset_embedding = nn.Embedding(n_datasets, n_atom_basis)

        # --- Electronic embeddings ---
        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings)

        # --- Filter network (same as vanilla PaiNN) ---
        self.share_filters = shared_filters
        if shared_filters:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        # --- Interaction + mixing blocks (same as vanilla PaiNN) ---
        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.n_atom_basis,
                activation=activation,
                epsilon=epsilon,
            ),
            self.n_interactions,
            shared_interactions,
        )

    def _get_conditioning_vector(
        self, dataset_id_per_atom: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute conditioning vector per atom.

        input mode    : returns [n_atoms, n_atom_basis] -- added to nuclear embedding.
        output_shift mode: returns [n_atoms, n_atom_basis] -- added to scalar_representation.
        multi_head    : returns None.
        """
        if self.dataset_embedding is None:
            return None
        return self.dataset_embedding(dataset_id_per_atom)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        idx_m = inputs[properties.idx_m]
        n_atoms = atomic_numbers.shape[0]

        dataset_id = inputs["dataset_id"].squeeze(-1)
        dataset_id_per_atom = dataset_id[idx_m]

        # --- pair features (same as vanilla PaiNN) ---
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # --- conditioning vector (None for multi_head) ---
        y = self._get_conditioning_vector(dataset_id_per_atom)

        # --- scalar embedding ---
        q = self.embedding(atomic_numbers)

        # input mode: inject dataset signal into scalar embedding
        if self.conditioning_mode == "input":
            q = q + y

        for emb in self.electronic_embeddings:
            q = q + emb(q, inputs)

        # PaiNN needs q with shape [n_atoms, 1, n_atom_basis]
        q = q.unsqueeze(1)

        # --- vector embedding init ---
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        # --- interaction + mixing blocks (vanilla PaiNN for all modes) ---
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)  # [n_atoms, n_atom_basis]

        # output_shift mode: inject dataset signal at scalar_representation
        if self.conditioning_mode == "output_shift":
            q = q + y

        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu

        return inputs

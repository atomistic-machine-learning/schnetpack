from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
from schnetpack import properties

__all__ = ["TimeConditioning"]


class TimeConditioning(nn.Module):
    """
    Conditions the representation on a per-atom scalar, e.g. the diffusion
    time of a generative model, or on a vector per atom or per structure,
    e.g. the parameters of a crystal's cell: the condition is embedded with
    a small MLP and added to the scalar representation. A per-structure
    condition is spread to the structure's atoms.

    Use as an output module placed *before* the prediction heads, so the
    heads see time-aware features.
    """

    # class defaults: modules saved before these options existed lack them
    n_in: int = 1
    per: Literal["atom", "structure"] = "atom"

    def __init__(
        self,
        n_atom_basis: int,
        condition_key: str = "t",
        n_in: int = 1,
        per: Literal["atom", "structure"] = "atom",
        n_hidden: int | None = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
    ):
        """
        Args:
            n_atom_basis: feature dimension of the scalar representation
            condition_key: input key holding the condition to embed
            n_in: number of components of the condition
            per: one condition per atom or per structure
            n_hidden: number of hidden units in the embedding MLP
                (default: n_atom_basis)
            n_layers: number of layers of the embedding MLP
            activation: activation function of the embedding MLP
        """
        super().__init__()
        if per not in ("atom", "structure"):
            raise ValueError(f"per must be 'atom' or 'structure', got {per!r}")
        self.condition_key = condition_key
        self.n_in = n_in
        self.per = per
        self.embedding = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_atom_basis,
            n_hidden=n_hidden or n_atom_basis,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = inputs[self.condition_key].reshape(-1, self.n_in)
        if self.per == "structure":
            condition = condition[inputs[properties.idx_m]]
        inputs["scalar_representation"] = inputs[
            "scalar_representation"
        ] + self.embedding(condition)
        return inputs

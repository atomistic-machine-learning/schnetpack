from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk

__all__ = ["TimeConditioning"]


class TimeConditioning(nn.Module):
    """
    Conditions the representation on a per-atom scalar, e.g. the diffusion
    time of a generative model: the scalar is embedded with a small MLP and
    added to the scalar representation.

    Use as an output module placed *before* the prediction heads, so the
    heads see time-aware features.
    """

    def __init__(
        self,
        n_atom_basis: int,
        condition_key: str = "t",
        n_hidden: Optional[int] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
    ):
        """
        Args:
            n_atom_basis: feature dimension of the scalar representation
            condition_key: input key holding the per-atom scalar to embed
            n_hidden: number of hidden units in the embedding MLP
                (default: n_atom_basis)
            n_layers: number of layers of the embedding MLP
            activation: activation function of the embedding MLP
        """
        super().__init__()
        self.condition_key = condition_key
        self.embedding = spk.nn.build_mlp(
            n_in=1,
            n_out=n_atom_basis,
            n_hidden=n_hidden or n_atom_basis,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        condition = inputs[self.condition_key].reshape(-1, 1)
        inputs["scalar_representation"] = inputs["scalar_representation"] + self.embedding(
            condition
        )
        return inputs

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from schnetpack.nn.residual_blocks import ResidualMLP

class ElectronicEmbedding(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with the
    electrons.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre_i (int):
            Number of residual blocks applied to atomic features in i branch
            (central atoms) before computing the interaction.
        num_residual_pre_j (int):
            Number of residual blocks applied to atomic features in j branch
            (neighbouring atoms) before computing the interaction.
        num_residual_post (int):
            Number of residual blocks applied to interaction features.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: str = "swish",
        is_charge: bool = False,
    ) -> None:
        """ Initializes the ElectronicEmbedding class. """
        super(ElectronicEmbedding, self).__init__()
        self.is_charge = is_charge
        self.linear_q = nn.Linear(num_features, num_features)
        if is_charge:  # charges are duplicated to use separate weights for +/-
            self.linear_k = nn.Linear(2, num_features, bias=False)
            self.linear_v = nn.Linear(2, num_features, bias=False)
        else:
            self.linear_k = nn.Linear(1, num_features, bias=False)
            self.linear_v = nn.Linear(1, num_features, bias=False)
        self.resblock = ResidualMLP(
            num_features,
            num_residual,
            activation=activation,
            zero_init=True,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        nn.init.orthogonal_(self.linear_k.weight)
        nn.init.orthogonal_(self.linear_v.weight)
        nn.init.orthogonal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        E: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # only for backwards compatibility
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = torch.zeros(x.size(0), dtype=torch.int64, device=x.device)
        q = self.linear_q(x)  # queries
        if self.is_charge:
            e = F.relu(torch.stack([E, -E], dim=-1))
        else:
            e = torch.abs(E).unsqueeze(-1)  # +/- spin is the same => abs
        enorm = torch.maximum(e, torch.ones_like(e))
        k = self.linear_k(e / enorm)[batch_seg]  # keys
        v = self.linear_v(e)[batch_seg]  # values
        dot = torch.sum(k * q, dim=-1) / k.shape[-1] ** 0.5  # scaled dot product
        a = nn.functional.softplus(dot)  # unnormalized attention weights
        anorm = a.new_zeros(num_batch).index_add_(0, batch_seg, a)
        if a.device.type == "cpu":  # indexing is faster on CPUs
            anorm = anorm[batch_seg]
        else:  # gathering is faster on GPUs
            anorm = torch.gather(anorm, 0, batch_seg)
        return self.resblock((a / (anorm + eps)).unsqueeze(-1) * v)
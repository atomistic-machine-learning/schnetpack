import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from schnetpack.nn.residual_blocks import ResidualMLP

class ElectronicEmbedding(nn.Module):
    """
    Single Head self attention block for updating atomic features through nonlocal interactions with the
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
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        E (FloatTensor [N]): 
            either charges or spin values per molecular graph
        num_batch (int): 
            number of molecular graphs in the batch
        batch_seq (LongTensor [N]): 
            segment ids (aka _idx_m) are used to separate different molecules in a batch
        eps (float): 
            small number to avoid division by zero
        """
        
        # queries (Batchsize x N_atoms, n_atom_basis)
        q = self.linear_q(x) 
        
        # to account for negative and positive charge
        if self.is_charge:
            e = F.relu(torch.stack([E, -E], dim=-1))
        # +/- spin is the same => abs
        else:
            e = torch.abs(E).unsqueeze(-1)  
        enorm = torch.maximum(e, torch.ones_like(e))

        # keys (Batchsize x N_atoms, n_atom_basis), the batch_seg ensures that the key is the same for all atoms belonging to the same graph
        k = self.linear_k(e / enorm)[batch_seg] 

        # values (Batchsize x N_atoms, n_atom_basis) the batch_seg ensures that the value is the same for all atoms belonging to the same graph
        v = self.linear_v(e)[batch_seg]

        # unnormalized, scaled attention weights, obtained by dot product of queries and keys (are logits)
        # scaling by square root of attention dimension
        weights = torch.sum(k * q, dim=-1) / k.shape[-1] ** 0.5

        # probability distribution of scaled unnormalized attention weights, by applying softmax function
        a = nn.functional.softplus(weights)

        # normalization factor for every molecular graph, by adding up attention weights of every atom in the graph
        anorm = a.new_zeros(num_batch).index_add_(0, batch_seg, a)
        
        # make tensor filled with anorm value at the position of the corresponding molecular graph, 
        # indexing faster on CPU, gather faster on GPU
        if a.device.type == "cpu": 
            anorm = anorm[batch_seg]
        else:
            anorm = torch.gather(anorm, 0, batch_seg)
        
        # return probability distribution of scaled normalized attention weights, eps is added for numerical stability (sum / batchsize equals 1)
        return self.resblock((a / (anorm + eps)).unsqueeze(-1) * v)
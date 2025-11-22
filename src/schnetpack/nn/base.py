from typing import Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_


__all__ = ["RMSNorm", "EquivariantRMSNorm", "Dense", "FeedForward"]

# RMSNorm from gemma
# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class RMSNorm(nn.Module):
   def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      add_unit_offset: bool = True,
   ):
      super().__init__()

      self.register_buffer("eps", torch.tensor(float(eps)))
      self.add_unit_offset = add_unit_offset
      self.weight = nn.Parameter(torch.zeros(dim))

   def forward(self, x):
      # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
      # See https://github.com/huggingface/transformers/pull/29402
      # This use x as float32 instead

      x_fp32 = x.float()

      variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
      out = x_fp32 * torch.rsqrt(variance + self.eps)
      out = out.to(self.weight.dtype)

      if self.add_unit_offset:
         out = out * (1 + self.weight)
      else:
         out = out * self.weight

      return out

class EquivariantRMSNorm(nn.Module):
    """
    RMSNorm that respects 3D vector geometry.
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, mu):
        # mu shape: [Batch, Atoms, 3, Features]
        mu_fp32 = mu.float()

        # Compute Squared Norm of vectors first (Sum over spatial dim 2)
        # This calculates (x^2 + y^2 + z^2) for every feature
        # Shape: [Batch, Atoms, Features]
        vector_squared_norms = mu_fp32.pow(2).sum(dim=-2) 

        # Calculate Mean of the Squared Norms across features
        # Shape: [Batch, Atoms, 1] (Broadcastable)
        variance = vector_squared_norms.mean(dim=-1, keepdim=True)

        # Create invariant scaler
        # We scale the vector by 1/RMS. This changes length, but not direction.
        # [Batch, Atoms, 1] -> [Batch, Atoms, 1, 1]
        inv_scale = torch.rsqrt(variance + self.eps).unsqueeze(-2)

        out = mu_fp32 * inv_scale
        out = out.to(self.weight.dtype)

        if self.add_unit_offset:
            out = out * (1 + self.weight)
        else:
            out = out * self.weight

        return out


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
        use_glu_variant: bool = True,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: number of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.use_glu_variant= use_glu_variant
        if self.use_glu_variant and activation is not None:
            out_features = out_features * 2

        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        if self.use_glu_variant and self.activation is not None:
            gate, y = F.linear(input, self.weight, self.bias).chunk(2, dim=-1)
            y = self.activation(gate) * y
        else:
            y = F.linear(input, self.weight, self.bias)
            y = self.activation(y)
        return y

class FeedForward(nn.Module):
    """
    Standard FeedForward Network (FFN) with optional GLU path.
    """
    def __init__(
        self, 
        n_atom_basis: int, 
        expansion_ratio: int = 2, 
        activation: Callable = F.silu,
        use_glu_variant: bool = True
    ):
        super().__init__()
        
        hidden_dim = n_atom_basis * expansion_ratio

        self.net_up = Dense(
            in_features=n_atom_basis, 
            out_features=hidden_dim, 
            bias=True, 
            activation=activation,
            use_glu_variant=use_glu_variant
        )

        self.net_down = Dense(
            in_features=hidden_dim, 
            out_features=n_atom_basis, 
            bias=True, 
            activation=None, # Linear output
            use_glu_variant=False
        )

    def forward(self, x):
        x = self.net_up(x)
        x = self.net_down(x)
        return x

# vi:ts=4 sw=4 et

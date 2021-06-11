import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
from typing import Tuple

__all__ = ["GatedEquivariantBlock"]


class GatedEquivariantBlock(nn.Module):
    def __init__(
        self,
        n_sin,
        n_vin,
        n_sout,
        n_vout,
        n_hidden,
        activation=F.silu,
        sactivation=None,
    ):
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = snn.Dense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            snn.Dense(n_sin + n_vout, n_hidden, activation=activation),
            snn.Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out

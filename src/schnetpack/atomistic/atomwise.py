import torch
import torch.nn as nn
from torch_scatter import segment_csr
from typing import Sequence, Union, Optional, Callable, Dict

import schnetpack as spk
import schnetpack.structure as structure


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: str = "sum",
        return_contributions: bool = False,
        custom_outnet: Callable = None,
        outnet_inputs: Union[str, Sequence[str]] = "scalar_representation",
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            aggregation_mode: one of {sum, avg} (default: sum)
            return_contributions: If true, returns also atomwise contributions.
            custom_outnet: Network used for atomistic outputs. Takes schnetpack input
                dictionary as input. Output is not normalized. If set to None,
                a pyramidal network is generated automatically.
            outnet_inputs: input dict entries to pass to outnet.
        """
        super(Atomwise, self).__init__()
        self.n_out = n_out

        # build output network
        self.outnet = custom_outnet or spk.nn.MLP(
            n_in=n_in, n_out=n_out, n_layers=2, activation=spk.nn.shifted_softplus
        )
        self.outnet_inputs = (
            [outnet_inputs] if isinstance(outnet_inputs, str) else outnet_inputs
        )

        self.aggregation_mode = aggregation_mode
        self.return_contributions = return_contributions

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # predict atomwise contributions
        outins = [inputs[k] for k in self.outnet_inputs]
        yi = self.outnet(*outins)

        # aggregate
        idx_m = inputs[structure.idx_m]
        tmp = torch.zeros((idx_m[-1] + 1, self.n_out), dtype=yi.dtype, device=yi.device)
        y = tmp.index_add(0, idx_m, yi)
        y = torch.squeeze(y, -1)

        if self.return_contributions:
            return y, yi
        return y

import torch
import torch.nn as nn
from torch_scatter import segment_csr
from typing import Sequence, Union, Optional, Callable

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
        mean: Optional[Union[torch.Tensor, float]] = None,
        stddev: Optional[Union[torch.Tensor, float]] = None,
        atomref: Optional[torch.Tensor] = None,
        custom_outnet: Callable = None,
        outnet_inputs: Union[str, Sequence[str]] = "scalar_representation",
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            aggregation_mode: one of {sum, avg} (default: sum)
            return_contributions: If true, returns also atomwise contributions.
            mean: Mean of property to predict.
                Should be specified per atom for `aggregation_mode=sum`, and exclude atomref, if given.
            stddev: Standard deviation of property to predict.
                Should be specified per atom for `aggregation_mode=sum`, and exclude atomref, if given.
            atomref: reference single-atom properties. Expects
                an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
                property of element Z.
            custom_outnet: Network used for atomistic outputs. Takes schnetpack input
                dictionary as input. Output is not normalized. If set to None,
                a pyramidal network is generated automatically.
            outnet_inputs: input dict entries to pass to outnet.
        """
        super(Atomwise, self).__init__()

        self.property = property
        self.return_contributions = return_contributions

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(atomref)
        else:
            self.atomref = None

        # build output network
        self.outnet = custom_outnet or spk.nn.MLP(
            n_in=n_in, n_out=n_out, n_layers=2, activation=nn.SiLU()
        )
        self.outnet_inputs = (
            [outnet_inputs] if isinstance(outnet_inputs, str) else outnet_inputs
        )

        # build standardization layer
        self.standardize = spk.nn.ScaleShift(mean, stddev)
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs):
        # predict atomwise contributions
        outins = [inputs[k] for k in self.outnet_inputs]
        yi = self.outnet(*outins)

        yi = self.standardize(yi)

        atomic_numbers = inputs[structure.Z]
        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        seg_m = inputs[structure.seg_m]
        y = segment_csr(yi, seg_m, reduce=self.aggregation_mode)
        y = torch.squeeze(y, -1)

        if self.return_contributions:
            return y, yi
        return y

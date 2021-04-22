import torch
import torch.nn as nn
from torch_scatter import segment_csr
from typing import Sequence, Union, Optional

import schnetpack as spk
import schnetpack.structure as structure


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    """

    def __init__(
        self,
        n_in: Union[int, Sequence],
        n_out: int = 1,
        aggregation_mode: str = "sum",
        n_layers: int = 2,
        n_neurons: Optional[int] = None,
        activation=spk.nn.shifted_softplus,
        return_contributions: bool = False,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
        representation_key="scalar_representation",
    ):
        super(Atomwise, self).__init__()

        if isinstance(n_in, Sequence):
            assert len(n_in) == 1, f"Atomwise only takes one input, but n_in is {n_in}"
            n_in = n_in[0]

        self.n_layers = n_layers
        self.property = property
        self.return_contributions = return_contributions
        self.representation_key = representation_key

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(atomref)
        else:
            self.atomref = None

        # build output network
        if outnet is None:
            self.out_net = spk.nn.MLP(n_in, n_out, n_neurons, n_layers, activation)
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = spk.nn.ScaleShift(mean, stddev)
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs):
        atomic_numbers = inputs[structure.Z]
        seg_m = inputs[structure.seg_m]
        representation = inputs[self.representation_key]

        # run prediction
        yi = self.out_net(representation)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = segment_csr(yi, seg_m, reduce=self.aggregation_mode)
        y = torch.squeeze(y, -1)

        if self.return_contributions:
            return y, yi
        return y

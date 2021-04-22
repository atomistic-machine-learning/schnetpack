import torch
import torch.nn as nn
from torch_scatter import segment_csr
from typing import Sequence, Union, Optional

import schnetpack as spk
import schnetpack.structure as structure


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
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
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            aggregation_mode: one of {sum, avg} (default: sum)
            n_layers: number of nn in output network (default: 2)
            n_neurons: number of neurons in each layer of the output
                network. If `None`, divide neurons by 2 in each layer. (default: None)
            activation: activation function for hidden nn
                (default: spk.nn.activations.shifted_softplus)
            return_contributions: If true, returns also atomwise contributions.
            mean: Mean of property to predict.
                Should be specified per atom for `aggregation_mode=sum`, and exclude atomref, if given.
            stddev: Standard deviation of property to predict.
                Should be specified per atom for `aggregation_mode=sum`, and exclude atomref, if given.
            atomref: reference single-atom properties. Expects
                an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
                property of element Z.
            outnet: Network used for atomistic outputs. Takes schnetpack input
                dictionary as input. Output is not normalized. If set to None,
                a pyramidal network is generated automatically.
            representation_key: The key of the representation to use in the provided input dictionary.
        """
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

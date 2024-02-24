from typing import Callable, Optional, Sequence, Union

import schnetpack as spk
import schnetpack.properties as properties
import torch
import torch.nn as nn
import torch.nn.functional as F


class WannierCenter(nn.Module):
    """
    Predict the Wannier center.

    The returned value is a vector + the position of the atom.

    Note, not all atoms have a Wannier center, only these whose wc_selector=True will
    be selected.


    # TODO, create a different key for the vector (see above)
    This is based on DipoleMoment, and the predicted vector will take the name set in
    `dipoles_key` and be stored in the inputs dictionary under that key.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        predict_magnitude: bool = False,
        dipole_key: str = properties.dipole_moment,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers
                resulting in a rectangular network.
                If None, the number of neurons is divided by two after each layer
                starting n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            predict_magnitude: If true, calculate magnitude of dipole
            dipole_key: the key under which the dipoles will be stored
        """
        super().__init__()

        self.dipole_key = dipole_key
        self.model_outputs = [dipole_key]
        self.predict_magnitude = predict_magnitude

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]
        _, atomic_dipoles = self.outnet((l0, l1))
        atomic_dipoles = torch.squeeze(atomic_dipoles, -1)

        y = positions + atomic_dipoles

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=False)

        wc_selector = inputs["wc_selector"].to(torch.bool)
        y = y[wc_selector, :]

        inputs[self.dipole_key] = y

        return inputs

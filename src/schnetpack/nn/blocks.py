from typing import Union, Sequence, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.nn as snn
from schnetpack.nn.activations import shifted_softplus

__all__ = ["build_mlp", "build_gated_equivariant_mlp"]


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    last_bias: bool = True,
    last_zero_init: bool = False,
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        snn.Dense(n_neurons[i], n_neurons[i + 1], activation=activation)
        for i in range(n_layers - 1)
    ]
    # assign a Dense layer (without activation function) to the output layer

    if last_zero_init:
        layers.append(
            snn.Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=last_bias,
            )
        )
    else:
        layers.append(
            snn.Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


def build_gated_equivariant_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_gating_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    sactivation: Callable = F.silu,
):
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s instead of dense layers.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: Activation function for gating function.
        sactivation: Activation function for scalar outputs. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    if n_gating_hidden is None:
        n_gating_hidden = n_neurons[:-1]
    elif type(n_gating_hidden) is int:
        n_gating_hidden = [n_gating_hidden] * n_layers
    else:
        n_gating_hidden = list(n_gating_hidden)

    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        snn.GatedEquivariantBlock(
            n_sin=n_neurons[i],
            n_vin=n_neurons[i],
            n_sout=n_neurons[i + 1],
            n_vout=n_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        snn.GatedEquivariantBlock(
            n_sin=n_neurons[-2],
            n_vin=n_neurons[-2],
            n_sout=n_neurons[-1],
            n_vout=n_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


class Residual(nn.Module):
    """
    Pre-activation residual block inspired by He, Kaiming, et al. "Identity
    mappings in deep residual networks."
    """

    def __init__(
        self,
        num_features: int,
        activation: Union[Callable, nn.Module] = None,
        bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        """
        Args:
            num_features: Dimensions of feature space.
            activation: activation function
        """
        super(Residual, self).__init__()
        # initialize attributes

        self.activation1 = activation  # (num_features)
        self.linear1 = nn.Linear(num_features, num_features, bias=bias)
        self.activation2 = activation  # (num_features)
        self.linear2 = nn.Linear(num_features, num_features, bias=bias)
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = True) -> None:
        """Initialize parameters to compute an identity mapping."""
        nn.init.orthogonal_(self.linear1.weight)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
        else:
            nn.init.orthogonal_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block to input atomic features.
        N: Number of atoms.
        num_features: Dimensions of feature space.

        Args:
            x (FloatTensor [N, num_features]):Input feature representations of atoms.

        Returns:
            y (FloatTensor [N, num_features]): Output feature representations of atoms.
        """
        y = self.activation1(x)
        y = self.linear1(y)
        y = self.activation2(y)
        y = self.linear2(y)
        return x + y


class ResidualStack(nn.Module):
    """
    Stack of num_blocks pre-activation residual blocks evaluated in sequence.
    """

    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: Union[Callable, nn.Module],
        bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        """
        Args:
            num_blocks: Number of residual blocks to be stacked in sequence.
            num_features: Dimensions of feature space.
            activation: activation function
        """
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList(
            [
                Residual(num_features, activation, bias, zero_init)
                for i in range(num_residual)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies all residual blocks to input features in sequence.
        N: Number of inputs.
        num_features: Dimensions of feature space.

        Args:
            x (FloatTensor [N, num_features]): Input feature representations.

        Returns:
            y (FloatTensor [N, num_features]):Output feature representations.
        """
        for residual in self.stack:
            x = residual(x)
        return x


class ResidualMLP(nn.Module):
    """Residual MLP with num_residual residual blocks."""

    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: Union[Callable, nn.Module],
        bias: bool = True,
        zero_init: bool = False,
    ):
        """
        Args:
            num_features: Dimensions of feature space.
            num_residual: Number of residual blocks to be stacked in sequence.
            activation: activation function
        """

        super(ResidualMLP, self).__init__()
        self.residual = ResidualStack(
            num_features, num_residual, activation=activation, bias=bias, zero_init=True
        )

        self.linear = nn.Linear(num_features, num_features, bias=bias)
        self.activation = activation
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = False) -> None:
        if zero_init:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.orthogonal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(self.residual(x)))

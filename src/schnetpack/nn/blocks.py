from typing import Union, Sequence, Callable, Optional

import torch
import torch.nn as nn
from schnetpack.nn import Dense

__all__ = ["MLP", "ElementwiseMLP"]


class MLP(nn.Module):
    """Multiple layer fully connected perceptron neural network."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = nn.SiLU(),
    ):
        """
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
        super(MLP, self).__init__()
        # get list of number of nodes in input, hidden & output layers
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = max(n_out, c_neurons // 2)
            self.n_neurons.append(n_out)
        else:
            # get list of number of nodes hidden layers
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            else:
                n_hidden = list(n_hidden)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Compute neural network output.

        Args:
            inputs (torch.Tensor): network input.

        Returns:
            torch.Tensor: network output.

        """
        return self.out_net(inputs)


class ElementwiseMLP(nn.Module):
    """Multiple layer fully connected perceptron neural network with on enetwork per element."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        elements: Union[int, Sequence[int]],
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = nn.SiLU(),
        trainable: bool = False,
    ):
        """
        Args:
            n_in: number of input nodes.
            n_out: number of output nodes.
            elements: list of atomic numbers of the elements present in the data.
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
        super(ElementwiseMLP, self).__init__()

        self.n_elements = len(elements)
        self.elements = elements
        self.z_max = int(max(elements))

        self.basic_mlps = nn.ModuleList(
            [
                MLP(
                    n_in,
                    n_out,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                    activation=activation,
                )
                for _ in range(self.n_elements)
            ]
        )

        elemental_weights = self._init_basis()
        self.element_mask = nn.Embedding.from_pretrained(
            elemental_weights, freeze=not trainable
        )

    def _init_basis(self):
        """
        Initialize a one hot mask for each element which can then be applied to the output MLPs.
        """
        elemental_weights = torch.zeros(self.z_max + 1, self.n_elements)
        for idx, element in enumerate(self.elements):
            elemental_weights[element, idx] = 1.0
        return elemental_weights

    def forward(self, atomic_numbers, representation):
        """Compute neural network output and mask according to element

        Args:
            atomic_numbers (torch.Tensor): atomic numbers.
            representation (torch.Tensor): representation of the molecular structure.

        Returns:
            torch.Tensor: network output.
        """
        mask = self.element_mask(atomic_numbers)
        y = torch.cat([net(representation) for net in self.basic_mlps], dim=1)
        y = torch.sum(y * mask, dim=1, keepdim=True)
        return y

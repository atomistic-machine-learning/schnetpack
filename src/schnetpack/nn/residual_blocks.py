import torch
import torch.nn as nn
import torch.nn.functional as F
from schnetpack.nn.activations import shifted_softplus

import math
from typing import Callable, Dict

class ShiftedSoftplus(nn.Module):
    """
    Shifted softplus activation function with learnable feature-wise parameters:
    f(x) = alpha/beta * (softplus(beta*x) - log(2))
    softplus(x) = log(exp(x) + 1)
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        num_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the softplus function.
        initial_beta (float):
            Initial "temperature" beta of the softplus function.
    """

    def __init__(
        self, num_features: int, initial_alpha: float = 1.0, initial_beta: float = 1.0
    ) -> None:
        """ Initializes the ShiftedSoftplus class. """
        super(ShiftedSoftplus, self).__init__()
        self._log2 = math.log(2)
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(num_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters alpha and beta. """
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, num_features]):
                Input features.

        Returns:
            y (FloatTensor [:, num_features]):
                Activated features.
        """
        return self.alpha * torch.where(
            self.beta != 0,
            (F.softplus(self.beta * x) - self._log2) / self.beta,
            0.5 * x,
        )


class Residual(nn.Module):
    """
    Pre-activation residual block inspired by He, Kaiming, et al. "Identity
    mappings in deep residual networks.".

    Arguments:
        num_features (int):
            Dimensions of feature space.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        num_features: int,
        activation: Callable = "ssp",
        bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        """ Initializes the Residual class. """
        super(Residual, self).__init__()
        # initialize attributes
        if activation == "ssp":
            Activation = ShiftedSoftplus

        self.activation1 = Activation(num_features)
        self.linear1 = nn.Linear(num_features, num_features, bias=bias)
        self.activation2 = Activation(num_features)
        self.linear2 = nn.Linear(num_features, num_features, bias=bias)
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = True) -> None:
        """ Initialize parameters to compute an identity mapping. """
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

        Arguments:
            x (FloatTensor [N, num_features]):
                Input feature representations of atoms.

        Returns:
            y (FloatTensor [N, num_features]):
                Output feature representations of atoms.
        """
        y = self.activation1(x)
        y = self.linear1(y)
        y = self.activation2(y)
        y = self.linear2(y)
        return x + y


class ResidualStack(nn.Module):
    """
    Stack of num_blocks pre-activation residual blocks evaluated in sequence.

    Arguments:
        num_blocks (int):
            Number of residual blocks to be stacked in sequence.
        num_features (int):
            Dimensions of feature space.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: str = "ssp",
        bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        """ Initializes the ResidualStack class. """
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

        Arguments:
            x (FloatTensor [N, num_features]):
                Input feature representations.

        Returns:
            y (FloatTensor [N, num_features]):
                Output feature representations.
        """
        for residual in self.stack:
            x = residual(x)
        return x
    

class ResidualMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: str = "ssp",
        bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super(ResidualMLP, self).__init__()
        self.residual = ResidualStack(
            num_features, num_residual, activation=activation, bias=bias, zero_init=True
        )
        # initialize activation function
        if activation == "ssp":
            self.activation = ShiftedSoftplus(num_features)

        self.linear = nn.Linear(num_features, num_features, bias=bias)
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
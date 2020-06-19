import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional
import types


__all__ = [
    "shifted_softplus",
    "Swish",
    "activation_factory",
    "get_activation_layer",
    "none_activation",
    "softplus_inverse",
]


def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - np.log(2.0)


class Swish(nn.Module):
    def __init__(self, n_features, initial_alpha=1.0, initial_beta=1.702):
        super(Swish, self).__init__()
        self.n_features = n_features
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(self.n_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(self.n_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x):
        return self.alpha * x * torch.sigmoid(self.beta * x)


def activation_factory(activation):
    """
    Wrapper to build activation function. Needed for Activations with trainable params.

    Args:
        activation (nn.Module): activation function

    Returns:
        (callable): factory function for activation functions
    """

    def factory(*args, **kwargs):
        return activation(*args, **kwargs)

    return factory


def get_activation_layer(activation, n_features=None):
    """
    Build activation layer.

    Args:
        activation (type or callable): activation function or class
        n_features (int, optional): feature dimension for trainable activations

    Returns:
        (callable): activation layer

    """
    # none activation:
    if activation is None:
        return none_activation
    # function type activation
    if isinstance(activation, types.FunctionType):
        return activation
    # swish activation
    if activation == Swish:
        return activation(n_features)
    # class type activation
    return activation()


def none_activation(x):
    """
    Placeholder function if activation is None.

    """
    return x


def softplus_inverse(x):
    """
    Inverse softplus transformation. This is useful for initialization of parameters
    that are constrained to be positive.

    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))

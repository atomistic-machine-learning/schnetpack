from typing import Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_


__all__ = ["Dense"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


#
# class ScaleShift(nn.Module):
#     r"""Scale and shift layer for standardization.
#
#     .. math::
#        y = x \times \sigma + \mu
#
#     """
#
#     def __init__(
#         self,
#         mean: Optional[torch.Tensor] = None,
#         stddev: Optional[torch.Tensor] = None,
#         trainable: bool = False,
#     ):
#         """
#         Args:
#             mean: mean value :math:`\mu`.
#             stddev: standard deviation value :math:`\sigma`.
#             trainable:
#         """
#         super(ScaleShift, self).__init__()
#
#         mean = mean or torch.tensor(0.0)
#         stddev = stddev or torch.tensor(1.0)
#         self.mean = nn.Parameter(mean, requires_grad=trainable)
#         self.stddev = nn.Parameter(stddev, requires_grad=trainable)
#
#     def forward(self, input):
#         """Compute layer output.
#
#         Args:
#             input (torch.Tensor): input data.
#
#         Returns:
#             torch.Tensor: layer output.
#
#         """
#         y = input * self.stddev + self.mean
#         return y

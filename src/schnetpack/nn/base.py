import torch
from torch import nn as nn
from torch.nn.init import xavier_uniform_

from schnetpack.nn.initializers import zeros_initializer


__all__ = ["Dense", "GetItem", "ScaleShift", "Standardize", "Aggregate"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        # initialize linear layer y = xW^T + b
        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y


class GetItem(nn.Module):
    """Extraction layer to get an item from SchNetPack dictionary of input tensors.

    Args:
        key (str): Property to be extracted from SchNetPack input tensors.

    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: layer output.

        """
        return inputs[self.key]


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class Standardize(nn.Module):
    """
    Standardization routine for NN close to input,
    e.g. symmetry functions. Values of mean and sttdev must
    be computed via pass over data set.

    Args:
        mean (torch.Tensor): Mean of data.
        stddev (torch.Tensor): Standard deviation of data.
        eps (float): Small offset to avoid zero division.
    """

    def __init__(self, mean, stddev, eps=1e-9):
        super(Standardize, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Transformed data.
        """
        # Add small number to catch divide by zero
        y = (input - self.mean) / (self.stddev + self.eps)
        return y


class Aggregate(nn.Module):
    """
    Pooling layer with optional masking

    Args:
        axis (int): pooling axis
        mean (bool): use average instead of sum pooling
    """

    def __init__(self, axis, mean=False, keepdim=True):
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim
        super(Aggregate, self).__init__()

    def forward(self, input, mask=None):
        """
        Args:
            input (torch.Tensor): Input tensor to be pooled.
            mask (torch.Tensor): Mask to be applied (e.g. neighbors mask)

        Returns:
            torch.Tensor: Pooled tensor.
        """
        if mask is not None:
            input = input * mask[..., None]
        y = torch.sum(input, self.axis)

        if self.average:
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N

        return y

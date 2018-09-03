import torch
from torch import nn as nn
from torch.nn.init import xavier_uniform_

from schnetpack.nn.initializers import zeros_initializer

__all__ = [
    'Dense', 'GetItem', 'ScaleShift', 'Standardize', 'Aggregate'
]


class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y


class GetItem(nn.Module):
    """
    Extracts a single item from the standard SchNetPack input dictionary.

    Args:
        key (str): Property to be extracted from SchNetPack input tensors.
    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, input):
        """
        Args:
            input (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Extracted item.
        """
        return input[self.key]


class ScaleShift(nn.Module):
    """
    Standardization layer encoding the standardization of output layers according to:
    :math:`X_\sigma = (X - \mu_X) / \sigma_X`

    Args:
        mean (torch.Tensor): Mean of data.
        stddev (torch.Tensor): Standard deviation of data.
    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('stddev', stddev)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Transformed data.
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
        self.register_buffer('mean', mean)
        self.register_buffer('stddev', stddev)
        self.register_buffer('eps', torch.ones_like(stddev) * eps)

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

    def __init__(self, axis, mean=False):
        self.average = mean
        self.axis = axis
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
                N = torch.sum(mask, self.axis, keepdim=True)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N

        return y

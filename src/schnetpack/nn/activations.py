import math
import torch

from torch.nn import functional

__all__ = ["shifted_softplus", "softplus_inverse"]


def shifted_softplus(x: torch.Tensor):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - math.log(2.0)


def softplus_inverse(x: torch.Tensor):
    """
    Inverse of the softplus function.

    Args:
        x (torch.Tensor): Input vector

    Returns:
        torch.Tensor: softplus inverse of input.
    """
    return x + (torch.log(-torch.expm1(-x)))

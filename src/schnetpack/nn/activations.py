import math
import torch

from torch.nn import functional

__all__ = ["shifted_softplus", "softplus_inverse", "ShiftedSoftplus"]


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


class ShiftedSoftplus(torch.nn.Module):
    """
    Shifted softplus activation function with learnable feature-wise parameters:
    f(x) = alpha/beta * (softplus(beta*x) - log(2))
    softplus(x) = log(exp(x) + 1)
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    With learnable parameters alpha and beta, the shifted softplus function can
    become equivalent to ReLU (if alpha is equal 1 and beta approaches infinity) or to
    the identity function (if alpha is equal 2 and beta is equal 0).
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            initial_alpha: Initial "scale" alpha of the softplus function.
            initial_beta: Initial "temperature" beta of the softplus function.
            trainable: If True, alpha and beta are trained during optimization.
        """
        super(ShiftedSoftplus, self).__init__()
        initial_alpha = torch.tensor(initial_alpha)
        initial_beta = torch.tensor(initial_beta)

        if trainable:
            self.alpha = torch.nn.Parameter(torch.FloatTensor([initial_alpha]))
            self.beta = torch.nn.Parameter(torch.FloatTensor([initial_beta]))
        else:
            self.register_buffer("alpha", initial_alpha)
            self.register_buffer("beta", initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Args:
            x (FloatTensor [:, num_features]): Input features.

        Returns:
            y (FloatTensor [:, num_features]): Activated features.
        """
        return self.alpha * torch.where(
            self.beta != 0,
            (torch.nn.functional.softplus(self.beta * x) - math.log(2)) / self.beta,
            0.5 * x,
        )

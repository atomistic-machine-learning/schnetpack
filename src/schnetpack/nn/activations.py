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

    Arguments:
        num_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the softplus function.
        initial_beta (float):
            Initial "temperature" beta of the softplus function.
    """

    def __init__(
        self, 
        num_features: int, 
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0
    ) -> None:
        """ Initializes the ShiftedSoftplus class. """
        super(ShiftedSoftplus, self).__init__()
        self._log2 = math.log(2)
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", torch.nn.Parameter(torch.Tensor(num_features)))
        self.register_parameter("beta", torch.nn.Parameter(torch.Tensor(num_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters alpha and beta. """
        torch.nn.init.constant_(self.alpha, self.initial_alpha)
        torch.nn.init.constant_(self.beta, self.initial_beta)

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
            (torch.nn.functional.F.softplus(self.beta * x) - self._log2) / self.beta,
            0.5 * x,
        )
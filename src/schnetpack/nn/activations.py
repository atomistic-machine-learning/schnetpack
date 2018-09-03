import numpy as np
from torch.nn import functional as F


def shifted_softplus(x):
    """
    Shifted softplus activation function of the form:
    :math:`y = ln( e^{-x} + 1 ) - ln(2)`

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shifted softplus applied to x

    """
    return F.softplus(x) - np.log(2.0)
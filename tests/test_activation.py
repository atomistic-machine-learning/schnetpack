
import torch
import numpy as np

from numpy.testing import assert_allclose
from schnetpack.nn.activations import shifted_softplus


def test_activation_softplus():
    # simple tensor
    x = torch.tensor([0.0, 1.0, 0.5, 2.0])
    assert_allclose(torch.log(1. + torch.exp(x)) - np.log(2), shifted_softplus(x))
    # random tensor
    torch.manual_seed(42)
    x = torch.randn((10, 5), dtype=torch.double)
    assert_allclose(torch.log(1. + torch.exp(x)) - np.log(2), shifted_softplus(x))
    x = 10 * torch.randn((10, 5), dtype=torch.double)
    assert_allclose(torch.log(1. + torch.exp(x)) - np.log(2), shifted_softplus(x))

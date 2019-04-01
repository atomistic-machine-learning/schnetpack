import torch
import numpy as np

from schnetpack.nn.activations import shifted_softplus


def test_activation_softplus():
    # simple tensor
    x = torch.tensor([0.0, 1.0, 0.5, 2.0])
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, shifted_softplus(x), atol=0.0, rtol=1.0e-7)
    # random tensor
    torch.manual_seed(42)
    x = torch.randn((10, 5), dtype=torch.double)
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, shifted_softplus(x), atol=0.0, rtol=1.0e-7)
    x = 10 * torch.randn((10, 5), dtype=torch.double)
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, shifted_softplus(x), atol=0.0, rtol=1.0e-7)

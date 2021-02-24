import torch
import numpy as np

import schnetpack as spk


def test_activation_softplus():
    # simple tensor
    x = torch.tensor([0.0, 1.0, 0.5, 2.0])
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, spk.nn.shifted_softplus(x), atol=0.0, rtol=1.0e-7)
    # random tensor
    torch.manual_seed(42)
    x = torch.randn((10, 5), dtype=torch.double)
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, spk.nn.shifted_softplus(x), atol=0.0, rtol=1.0e-7)
    x = 10 * torch.randn((10, 5), dtype=torch.double)
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, spk.nn.shifted_softplus(x), atol=0.0, rtol=1.0e-7)


def test_shape_ssp():
    in_data = torch.rand(10)
    out_data = spk.nn.shifted_softplus(in_data)
    assert in_data.shape == out_data.shape

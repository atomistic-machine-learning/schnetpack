import torch

from schnetpack.nn.initializers import zeros_initializer


def test_zeros_initializer():
    # tensor of all ones
    res = zeros_initializer(torch.ones(2, 3))
    assert torch.allclose(torch.zeros(2, 3), res, atol=0.0, rtol=1.0e-7)
    # random tensor with elements in [0, 1)
    torch.manual_seed(99)
    res = zeros_initializer(torch.rand((10, 5, 15), dtype=torch.float))
    assert torch.allclose(torch.zeros(10, 5, 15), res, atol=0.0, rtol=1.0e-7)
    # random tensor
    res = zeros_initializer(5.2 * torch.rand(10, dtype=torch.float))
    assert torch.allclose(torch.zeros(10), res, atol=0.0, rtol=1.0e-7)
    res = zeros_initializer(7.6 * torch.rand((2, 100), dtype=torch.float) + 2.5)
    assert torch.allclose(torch.zeros(2, 100), res, atol=0.0, rtol=1.0e-7)

import torch
import numpy as np

from schnetpack.nn.cutoff import CosineCutoff, MollifierCutoff


def test_cutoff_cosine():
    # cosine cutoff with radius 1.8
    cutoff = CosineCutoff(cutoff=1.8)
    # check cutoff radius
    assert abs(1.8 - cutoff.cutoff) < 1.0e-12
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((10, 5, 20), dtype=torch.float)
    # check cutoff values
    expt = 0.5 * (1.0 + torch.cos(dist * np.pi / 1.8))
    assert torch.allclose(expt, cutoff(dist), atol=0.0, rtol=1.0e-7)
    # compute expected values for 3.5 times distance
    values = 0.5 * (1.0 + torch.cos(3.5 * dist * np.pi / 1.8))
    values[3.5 * dist >= 1.8] = 0.0
    assert torch.allclose(values, cutoff(3.5 * dist), atol=0.0, rtol=1.0e-7)


def test_cutoff_mollifier():
    # mollifier cutoff with radius 2.3
    cutoff = MollifierCutoff(cutoff=2.3)
    # check cutoff radius
    assert abs(2.3 - cutoff.cutoff) < 1.0e-12
    # tensor of zeros
    dist = torch.zeros((4, 1, 1))
    assert torch.allclose(torch.ones(4, 1, 1), cutoff(dist), atol=0.0, rtol=1.0e-7)
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((1, 3, 9), dtype=torch.float)
    # check cutoff values
    expt = torch.exp(1.0 - 1.0 / (1.0 - (dist / 2.3) ** 2))
    assert torch.allclose(expt, cutoff(dist), atol=0.0, rtol=1.0e-7)
    # compute cutoff values and expected values
    comp = cutoff(3.8 * dist)
    expt = torch.exp(1.0 - 1.0 / (1.0 - (3.8 * dist / 2.3) ** 2))
    expt[3.8 * dist >= 2.3] = 0.0
    assert torch.allclose(expt, comp, atol=0.0, rtol=1.0e-7)

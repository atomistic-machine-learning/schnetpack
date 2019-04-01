import torch
import numpy as np

from schnetpack.nn.cutoff import CosineCutoff, MollifierCutoff, HardCutoff


def test_cutoff_cosine_default():
    # cosine cutoff with default radius
    cutoff = CosineCutoff()
    # check cutoff radius
    assert abs(5.0 - cutoff.cutoff) < 1.0e-12
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((1, 15), dtype=torch.float)
    # check cutoff values
    expt = 0.5 * (1.0 + torch.cos(dist * np.pi / 5.0))
    assert torch.allclose(expt, cutoff(dist), atol=0.0, rtol=1.0e-7)
    expt = 0.5 * (1.0 + torch.cos(2.0 * dist * np.pi / 5.0))
    assert torch.allclose(expt, cutoff(2.0 * dist), atol=0.0, rtol=1.0e-7)
    expt = 0.5 * (1.0 + torch.cos(4.0 * dist * np.pi / 5.0))
    assert torch.allclose(expt, cutoff(4.0 * dist), atol=0.0, rtol=1.0e-7)
    # compute cutoff values and expected values
    comp = cutoff(5.5 * dist)
    expt = 0.5 * (1.0 + torch.cos(5.5 * dist * np.pi / 5.0))
    expt[5.5 * dist >= 5.0] = 0.0
    assert torch.allclose(expt, comp, atol=0.0, rtol=1.0e-7)


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


def test_cutoff_mollifier_default():
    # mollifier cutoff with default radius
    cutoff = MollifierCutoff()
    # check cutoff radius
    assert abs(5.0 - cutoff.cutoff) < 1.0e-12
    # tensor of zeros
    dist = torch.zeros((5, 2, 3))
    assert torch.allclose(torch.ones(5, 2, 3), cutoff(dist), atol=0.0, rtol=1.0e-7)
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((20, 1), dtype=torch.float)
    # check cutoff values
    expt = torch.exp(1.0 - 1.0 / (1.0 - (dist / 5.0) ** 2))
    assert torch.allclose(expt, cutoff(dist), atol=0.0, rtol=1.0e-7)
    # compute cutoff values and expected values
    comp = cutoff(6.0 * dist)
    expt = torch.exp(1.0 - 1.0 / (1.0 - (6.0 * dist / 5.0) ** 2))
    expt[6.0 * dist >= 5.0] = 0.0
    assert torch.allclose(expt, comp, atol=0.0, rtol=1.0e-7)


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


def test_cutoff_hard_default():
    # hard cutoff with default radius
    cutoff = HardCutoff()
    # check cutoff radius
    assert abs(5.0 - cutoff.cutoff) < 1.0e-12
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((1, 20), dtype=torch.float)
    # compute expected values
    def expt(distances):
        res = torch.ones_like(distances, dtype=torch.float)
        res[distances > 5.0] = 0.0
        return res

    # check cutoff values
    assert torch.allclose(expt(dist), cutoff(dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(0.5 * dist), cutoff(0.5 * dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(3.5 * dist), cutoff(3.5 * dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(4.9 * dist), cutoff(4.9 * dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(5.0 * dist), cutoff(5.0 * dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(7.5 * dist), cutoff(7.5 * dist), atol=0.0, rtol=1.0e-7)


def test_cutoff_hard():
    # hard cutoff with radius 2.5
    cutoff = HardCutoff(cutoff=2.5)
    # check cutoff radius
    assert abs(2.5 - cutoff.cutoff) < 1.0e-12
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((30, 20, 10), dtype=torch.float)
    # compute expected values
    def expt(distances):
        res = torch.ones_like(distances, dtype=torch.float)
        res[distances > 2.5] = 0.0
        return res

    # check cutoff values
    assert torch.allclose(expt(dist), cutoff(dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(0.5 * dist), cutoff(0.5 * dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(2.2 * dist), cutoff(2.2 * dist), atol=0.0, rtol=1.0e-7)
    assert torch.allclose(expt(3.7 * dist), cutoff(3.7 * dist), atol=0.0, rtol=1.0e-7)

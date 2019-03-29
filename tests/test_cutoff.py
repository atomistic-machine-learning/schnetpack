
import torch
import numpy as np

from numpy.testing import assert_allclose
from schnetpack.nn.cutoff import CosineCutoff, MollifierCutoff, HardCutoff


def test_cutoff_cosine_default():
    # cosine cutoff with default radius
    cutoff = CosineCutoff()
    # check cutoff radius
    assert_allclose(5.0, cutoff.cutoff)
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((1, 15), dtype=torch.float)
    # check cutoff values
    assert_allclose(0.5 * (1. + torch.cos(dist * np.pi / 5.)), cutoff(dist))
    assert_allclose(0.5 * (1. + torch.cos(2. * dist * np.pi / 5.)), cutoff(2. * dist))
    assert_allclose(0.5 * (1. + torch.cos(4. * dist * np.pi / 5.)), cutoff(4. * dist))
    # compute cutoff values and expected values
    comp = cutoff(5.5 * dist)
    expt = 0.5 * (1. + torch.cos(5.5 * dist * np.pi / 5.))
    expt[5.5 * dist >= 5.0] = 0.
    assert_allclose(expt, comp)


def test_cutoff_cosine():
    # cosine cutoff with radius 1.8
    cutoff = CosineCutoff(cutoff=1.8)
    # check cutoff radius
    assert_allclose(1.8, cutoff.cutoff)
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((10, 5, 20), dtype=torch.float)
    # check cutoff values
    assert_allclose(0.5 * (1. + torch.cos(dist * np.pi / 1.8)), cutoff(dist))
    # compute expected values for 3.5 times distance
    values = 0.5 * (1. + torch.cos(3.5 * dist * np.pi / 1.8))
    values[3.5 * dist >= 1.8] = 0.
    assert_allclose(values, cutoff(3.5 * dist))


def test_cutoff_mollifier_default():
    # mollifier cutoff with default radius
    cutoff = MollifierCutoff()
    # check cutoff radius
    assert_allclose(5.0, cutoff.cutoff)
    # tensor of zeros
    dist = torch.zeros((5, 2, 3))
    assert_allclose(torch.ones(5, 2, 3), cutoff(dist))
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((20, 1), dtype=torch.float)
    # check cutoff values
    assert_allclose(torch.exp(1. - 1. / (1. - (dist / 5.)**2)), cutoff(dist))
    # compute cutoff values and expected values
    comp = cutoff(6. * dist)
    expt = torch.exp(1. - 1. / (1. - (6. * dist / 5.)**2))
    expt[6. * dist >= 5.0] = 0.
    assert_allclose(expt, comp)


def test_cutoff_mollifier():
    # mollifier cutoff with radius 2.3
    cutoff = MollifierCutoff(cutoff=2.3)
    # check cutoff radius
    assert_allclose(2.3, cutoff.cutoff)
    # tensor of zeros
    dist = torch.zeros((4, 1, 1))
    assert_allclose(torch.ones(4, 1, 1), cutoff(dist))
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((1, 3, 9), dtype=torch.float)
    # check cutoff values
    assert_allclose(torch.exp(1. - 1. / (1. - (dist / 2.3)**2)), cutoff(dist))
    # compute cutoff values and expected values
    comp = cutoff(3.8 * dist)
    expt = torch.exp(1. - 1. / (1. - (3.8 * dist / 2.3)**2))
    expt[3.8 * dist >= 2.3] = 0.
    assert_allclose(expt, comp)


def test_cutoff_hard_default():
    # hard cutoff with default radius
    cutoff = HardCutoff()
    # check cutoff radius
    assert_allclose(5.0, cutoff.cutoff)
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((1, 20), dtype=torch.float)
    # check cutoff values
    assert_allclose(dist, cutoff(dist))
    assert_allclose(0.5 * dist, cutoff(0.5 * dist))
    assert_allclose(3.5 * dist, cutoff(3.5 * dist))
    assert_allclose(4.9 * dist, cutoff(4.9 * dist))
    assert_allclose(5.0 * dist, cutoff(5.0 * dist))
    # compute cutoff values and expected values
    comp = cutoff(7.5 * dist)
    expt = 7.5 * dist
    expt[expt >= 5.0] = 0.
    assert_allclose(expt, comp)


def test_cutoff_hard():
    # hard cutoff with radius 2.5
    cutoff = HardCutoff(cutoff=2.5)
    # check cutoff radius
    assert_allclose(2.5, cutoff.cutoff)
    # random tensor with elements in [0, 1)
    torch.manual_seed(42)
    dist = torch.rand((30, 20, 10), dtype=torch.float)
    # check cutoff values
    assert_allclose(dist, cutoff(dist))
    assert_allclose(0.5 * dist, cutoff(0.5 * dist))
    assert_allclose(2.2 * dist, cutoff(2.2 * dist))
    # compute cutoff values and expected values
    comp = cutoff(3.7 * dist)
    expt = 3.7 * dist
    expt[expt >= 2.5] = 0.
    assert_allclose(expt, comp)

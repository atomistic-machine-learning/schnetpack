
import torch

from numpy.testing import assert_allclose
from schnetpack.nn.cutoff import HardCutoff


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

import torch

from schnetpack.nn.base import Aggregate


def test_nn_aggregate_axis():
    data = torch.ones((1, 5, 4, 3), dtype=torch.float)
    agg = Aggregate(axis=0, mean=False)
    assert torch.allclose(torch.ones((5, 4, 3)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []
    agg = Aggregate(axis=1, mean=False)
    assert torch.allclose(5 * torch.ones((1, 4, 3)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []
    agg = Aggregate(axis=2, mean=False)
    assert torch.allclose(4 * torch.ones((1, 5, 3)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []
    agg = Aggregate(axis=3, mean=False)
    assert torch.allclose(3 * torch.ones((1, 5, 4)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []


def test_nn_aggregate_axis_mean():
    data = torch.ones((1, 5, 4, 3), dtype=torch.float)
    agg = Aggregate(axis=0, mean=True)
    assert torch.allclose(torch.ones((5, 4, 3)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []
    agg = Aggregate(axis=1, mean=True)
    assert torch.allclose(torch.ones((1, 4, 3)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []
    agg = Aggregate(axis=2, mean=True)
    assert torch.allclose(torch.ones((1, 5, 3)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []
    agg = Aggregate(axis=3, mean=True)
    assert torch.allclose(torch.ones((1, 5, 4)), agg(data), atol=0.0, rtol=1.0e-7)
    assert list(agg.parameters()) == []

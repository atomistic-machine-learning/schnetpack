import torch

from schnetpack.nn.base import Aggregate, MaxAggregate, SoftmaxAggregate


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


def test_nn_aggregate_axis_max():
    data = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])

    # Test different axes
    agg = MaxAggregate(axis=0)
    assert torch.allclose(torch.FloatTensor([2, 1, 2]), agg(data))
    agg = MaxAggregate(axis=1)
    assert torch.allclose(torch.FloatTensor([2, 2]), agg(data))

    # Test with a mask
    mask = torch.IntTensor([[1, 1, 0], [1, 0, 0]])
    agg = MaxAggregate(axis=0)
    assert torch.allclose(torch.FloatTensor([2, 1, 0]), agg(data, mask))
    agg = MaxAggregate(axis=1)
    assert torch.allclose(torch.FloatTensor([1, 2]), agg(data, mask))


def test_nn_aggregate_axis_softmax():
    data = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])

    # Test different axes
    agg = SoftmaxAggregate(axis=0)
    assert torch.allclose(
        torch.FloatTensor([1.76159415595576, 1, 1.76159415595576]), agg(data)
    )
    agg = SoftmaxAggregate(axis=1)
    assert torch.allclose(torch.FloatTensor([1.57521038260444] * 2), agg(data))

    # Test with a mask
    mask = torch.IntTensor([[1, 1, 0], [1, 0, 0]])
    agg = SoftmaxAggregate(axis=1)
    assert torch.allclose(torch.FloatTensor([0.731058578630005, 2]), agg(data, mask))

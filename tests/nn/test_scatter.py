import torch

from schnetpack.nn import scatter_add


def test_scatter_add_sums_over_indices():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx_i = torch.tensor([0, 2, 0])

    expected = torch.tensor([[6.0, 8.0], [0.0, 0.0], [3.0, 4.0]])
    assert torch.equal(scatter_add(x, idx_i, dim_size=3), expected)


def test_scatter_add_can_be_scripted():
    x, idx_i = torch.ones(4), torch.tensor([0, 0, 1, 1])

    scripted = torch.jit.script(scatter_add)
    assert torch.equal(scripted(x, idx_i, 2), scatter_add(x, idx_i, 2))

import pytest
import torch

import schnetpack.properties as structure
from schnetpack.atomistic import Atomwise


@pytest.fixture
def dummy_inputs():
    """Two molecules with 3 and 5 atoms, with a constant representation."""
    n_atoms = torch.tensor([3, 5])
    idx_m = torch.repeat_interleave(torch.arange(len(n_atoms)), repeats=n_atoms, dim=0)
    return {
        structure.n_atoms: n_atoms,
        structure.idx_m: idx_m,
        "scalar_representation": torch.randn(int(n_atoms.sum()), 8),
    }


@pytest.mark.parametrize("n_out", [1, 4])
def test_atomwise_avg_shape(dummy_inputs, n_out):
    """`avg` must work for vector-valued outputs, not just n_out == 1."""
    model = Atomwise(n_in=8, n_out=n_out, aggregation_mode="avg")
    y = model(dummy_inputs)["y"]

    expected = (2,) if n_out == 1 else (2, n_out)
    assert y.shape == torch.Size(expected)


@pytest.mark.parametrize("n_out", [1, 4])
def test_atomwise_avg_equals_sum_over_n_atoms(dummy_inputs, n_out):
    """`avg` is the `sum` aggregation divided by the number of atoms."""
    model = Atomwise(n_in=8, n_out=n_out, aggregation_mode="sum")
    y_sum = model(dict(dummy_inputs))["y"]

    model.aggregation_mode = "avg"
    y_avg = model(dict(dummy_inputs))["y"]

    n_atoms = dummy_inputs[structure.n_atoms]
    if n_out > 1:
        n_atoms = n_atoms.unsqueeze(-1)
    assert torch.allclose(y_avg, y_sum / n_atoms)

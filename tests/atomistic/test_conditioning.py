import pytest
import torch

import schnetpack.properties as structure
from schnetpack.model import TimeConditioning


@pytest.fixture
def inputs():
    """Two structures with 3 and 5 atoms, with a zero representation."""
    n_atoms = torch.tensor([3, 5])
    return {
        structure.n_atoms: n_atoms,
        structure.idx_m: torch.repeat_interleave(torch.arange(2), n_atoms),
        "scalar_representation": torch.zeros(8, 16),
        "t": torch.rand(8),
        "y": torch.randn(2, 6),
    }


def test_a_per_atom_scalar_is_added_to_every_atom(inputs):
    module = TimeConditioning(16, condition_key="t")
    out = module(dict(inputs))
    expected = module.embedding(inputs["t"][:, None])
    torch.testing.assert_close(out["scalar_representation"], expected)


def test_a_per_structure_vector_is_spread_to_its_atoms(inputs):
    module = TimeConditioning(16, condition_key="y", n_in=6, per="structure")
    out = module(dict(inputs))["scalar_representation"]
    assert out.shape == (8, 16)
    expected = module.embedding(inputs["y"])
    torch.testing.assert_close(out[:3], expected[0].expand(3, 16))
    torch.testing.assert_close(out[3:], expected[1].expand(5, 16))


def test_conditioning_refuses_an_unknown_row_layout():
    with pytest.raises(ValueError, match="per must be"):
        TimeConditioning(16, per="molecule")


def test_a_module_saved_before_the_vector_options_still_runs(inputs):
    module = TimeConditioning(16, condition_key="t")
    del module.n_in, module.per  # as unpickled from an older checkpoint
    out = module(dict(inputs))["scalar_representation"]
    torch.testing.assert_close(out, module.embedding(inputs["t"][:, None]))

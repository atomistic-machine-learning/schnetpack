from schnetpack.data.transforms import *
import numpy as np
import pytest


def assert_consistent(orig, transformed):
    for k, v in orig.items():
        assert (v == transformed[k]).all(), f"Changed value: {k}"


@pytest.fixture
def aseneighbor_list(cutoff):
    return ASENeighborList(cutoff)


def test_single_atom(single_atom, aseneighbor_list):
    props_after = aseneighbor_list(single_atom)
    assert_consistent(single_atom, props_after)
    assert len(props_after[Structure.Rij]) == 0
    assert len(props_after[Structure.idx_i]) == 0
    assert len(props_after[Structure.idx_j]) == 0
    assert len(props_after[Structure.cell_offset]) == 0


def test_cast(single_atom):
    allf64 = [k for k, v in single_atom.items() if v.dtype is torch.float64]
    other_types = {
        k: v.dtype for k, v in single_atom.items() if v.dtype is not torch.float64
    }

    assert len(allf64) > 0, single_atom
    props_after = CastTo32()(single_atom)

    for k in props_after:
        if k in allf64:
            assert props_after[k].dtype is torch.float32
        else:
            assert props_after[k].dtype is other_types[k]

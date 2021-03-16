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
    ats = single_atom
    props = {
        Structure.Z: ats.numbers,
        Structure.R: ats.positions,
        Structure.cell: ats.cell,
        Structure.pbc: ats.pbc,
    }
    props_after = aseneighbor_list(props)
    assert_consistent(props, props_after)
    assert len(props_after[Structure.Rij]) == 0
    assert len(props_after[Structure.idx_i]) == 0
    assert len(props_after[Structure.idx_j]) == 0
    assert len(props_after[Structure.cell_offset]) == 0


def test_cast(single_atom):
    ats = single_atom
    props = {
        Structure.Z: ats.numbers,
        Structure.R: ats.positions,
        Structure.cell: ats.cell,
        Structure.pbc: ats.pbc,
    }
    allf64 = [k for k, v in props.items() if v.dtype is torch.float64]
    other_types = {k: v.dtype for k, v in props.items() if v.dtype is not torch.float64}

    props_after = CastTo32()(props)

    for k in props_after:
        if k in allf64:
            assert props_after[k].dtype is torch.float32
        else:
            assert props_after[k].dtype is other_types[k]

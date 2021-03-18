from schnetpack.data.transforms import *
import numpy as np
import pytest
import torch


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


def atms2dict(atms):
    return {
        Structure.position: torch.tensor(atms.positions),
        Structure.Z: torch.tensor(atms.numbers),
    }


def test_remove_com(four_atoms):
    props = atms2dict(four_atoms)
    positions_trans = SubtractCenterOfMass()(props)

    com = torch.tensor([0.0, 0.0, 0.0])
    for r_i, m_i in zip(positions_trans[Structure.position], four_atoms.get_masses()):
        com += r_i * m_i

    torch.testing.assert_allclose(com, torch.tensor([0.0, 0.0, 0.0]))


def test_remove_cog(four_atoms):
    props = atms2dict(four_atoms)
    positions_trans = SubtractCenterOfGeometry()(props)

    com = torch.tensor([0.0, 0.0, 0.0])
    for r_i, m_i in zip(positions_trans[Structure.position], four_atoms.get_masses()):
        com += r_i

    torch.testing.assert_allclose(com, torch.tensor([0.0, 0.0, 0.0]))

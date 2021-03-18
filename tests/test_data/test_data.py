import pytest
import torch
from schnetpack.data import *
import os
from schnetpack import Structure


@pytest.fixture
def asedbpath(tmpdir):
    return os.path.join(tmpdir, "test.db")


@pytest.fixture(scope="function")
def asedb(asedbpath, example_data, property_shapes):
    available_props = list(property_shapes.keys())

    asedb = ASEAtomsData.create(
        datapath=asedbpath, available_properties=available_props
    )

    atoms_list, prop_list = zip(*example_data)
    asedb.add_systems(property_list=prop_list, atoms_list=atoms_list)
    yield asedb

    os.remove(asedb.datapath)
    del asedb


def test_asedb(asedb, example_data):
    assert os.path.exists(asedb.datapath)
    assert len(example_data) == len(asedb)
    assert asedb.metadata["_available_properties"] == asedb.available_properties

    props = asedb[0]
    assert set(props.keys()) == set(
        [
            Structure.Z,
            Structure.R,
            Structure.cell,
            Structure.pbc,
            Structure.n_atoms,
            Structure.idx,
        ]
        + asedb.available_properties
    )

    load_properties = asedb.available_properties[0:2]
    asedb.load_properties = load_properties
    props = asedb[0]
    assert set(props.keys()) == set(
        [
            Structure.Z,
            Structure.R,
            Structure.cell,
            Structure.pbc,
            Structure.n_atoms,
            Structure.idx,
        ]
        + load_properties
    )

    asedb.load_structure = False
    props = asedb[0]
    assert set(props.keys()) == set(
        [
            Structure.n_atoms,
            Structure.idx,
        ]
        + load_properties
    )

    asedb.update_metadata(test=1)
    assert asedb.metadata["test"] == 1


def test_asedb_getprops(asedb):
    props = asedb.get_properties(0)[0]
    assert set(props.keys()) == set(
        [
            Structure.Z,
            Structure.R,
            Structure.cell,
            Structure.pbc,
            Structure.n_atoms,
            Structure.idx,
        ]
        + asedb.available_properties
    )


def test_asedb_add(asedb, example_data):
    l = len(asedb)

    at, props = example_data[0]
    asedb.add_system(atoms=at, **props)

    props.update(
        {
            Structure.Z: at.numbers,
            Structure.R: at.positions,
            Structure.cell: at.cell,
            Structure.pbc: at.pbc,
        }
    )
    asedb.add_system(**props)

    p1 = asedb[l]
    p2 = asedb[l + 1]
    for k, v in p1.items():
        if k != "_idx":
            assert type(v) == torch.Tensor, k
            assert (p2[k] == v).all(), v

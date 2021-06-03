import pytest
import torch
import numpy as np
from schnetpack.data import *
import os
import schnetpack.properties as structure
from schnetpack.data import calculate_stats, AtomsLoader


@pytest.fixture
def asedbpath(tmpdir):
    return os.path.join(tmpdir, "test.db")


@pytest.fixture(scope="function")
def asedb(asedbpath, example_data, property_units):

    asedb = ASEAtomsData.create(
        datapath=asedbpath, distance_unit="A", property_unit_dict=property_units
    )

    atoms_list, prop_list = zip(*example_data)
    asedb.add_systems(property_list=prop_list, atoms_list=atoms_list)
    yield asedb

    os.remove(asedb.datapath)
    del asedb


def test_asedb(asedb, example_data):
    assert os.path.exists(asedb.datapath)
    assert len(example_data) == len(asedb)
    assert set(asedb.metadata["_property_unit_dict"].keys()) == set(
        asedb.available_properties
    )
    assert asedb.metadata["_property_unit_dict"] == asedb.units

    props = asedb[0]
    assert set(props.keys()) == set(
        [
            structure.Z,
            structure.R,
            structure.cell,
            structure.pbc,
            structure.n_atoms,
            structure.idx,
        ]
        + asedb.available_properties
    )

    load_properties = asedb.available_properties[0:2]
    asedb.load_properties = load_properties
    props = asedb[0]
    assert set(props.keys()) == set(
        [
            structure.Z,
            structure.R,
            structure.cell,
            structure.pbc,
            structure.n_atoms,
            structure.idx,
        ]
        + load_properties
    )

    asedb.load_structure = False
    props = asedb[0]
    assert set(props.keys()) == set(
        [
            structure.n_atoms,
            structure.idx,
        ]
        + load_properties
    )

    asedb.update_metadata(test=1)
    assert asedb.metadata["test"] == 1


def test_asedb_getprops(asedb):
    props = list(asedb.iter_properties(0))[0]
    assert set(props.keys()) == set(
        [
            structure.Z,
            structure.R,
            structure.cell,
            structure.pbc,
            structure.n_atoms,
            structure.idx,
        ]
        + asedb.available_properties
    )


def test_stats():
    data = []
    for i in range(6):
        Z = torch.tensor([1, 1, 1])
        off = 1.0 if i % 2 == 0 else -1.0
        d = {
            structure.Z: Z,
            structure.n_atoms: torch.tensor([len(Z)]),
            "property1": torch.tensor([(1.0 + len(Z) * off)]),
            "property2": torch.tensor([off]),
        }
        data.append(d)

    atomref = {"property1": torch.ones((100,)) / 3.0}
    for bs in range(1, 7):
        stats = calculate_stats(
            AtomsLoader(data, batch_size=bs),
            {"property1": True, "property2": False},
            atomref=atomref,
        )
        assert np.allclose(stats["property1"][0].numpy(), np.array([0.0]))
        assert np.allclose(stats["property1"][1].numpy(), np.array([1.0]))
        assert np.allclose(stats["property2"][0].numpy(), np.array([0.0]))
        assert np.allclose(stats["property2"][1].numpy(), np.array([1.0]))


def test_asedb_add(asedb, example_data):
    l = len(asedb)

    at, props = example_data[0]
    asedb.add_system(atoms=at, **props)

    props.update(
        {
            structure.Z: at.numbers,
            structure.R: at.positions,
            structure.cell: at.cell,
            structure.pbc: at.pbc,
        }
    )
    asedb.add_system(**props)

    p1 = asedb[l]
    p2 = asedb[l + 1]
    for k, v in p1.items():
        if k != "_idx":
            assert type(v) == torch.Tensor, k
            assert (p2[k] == v).all(), v

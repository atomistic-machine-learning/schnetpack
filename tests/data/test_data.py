import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from ase.db import connect

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData, calculate_stats


@pytest.fixture
def asedbpath(tmpdir):
    return os.path.join(tmpdir, "test.db")


@pytest.fixture(scope="function")
def asedb(asedbpath, example_data, property_units):
    with connect(asedbpath, use_lock_file=False) as conn:
        conn.metadata = {
            "_property_unit_dict": property_units,
            "_distance_unit": "Ang",
            "atomrefs": {},
        }

        for atoms, props in example_data:
            conn.write(atoms, data=props)

    asedb = ASEAtomsData(datapath=asedbpath, distance_unit="Ang")

    yield asedb

    if os.path.exists(asedbpath):
        os.remove(asedbpath)
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


def test_asedb_reads_from_threads(asedb):
    """Each thread reads through its own held connection."""
    expected = [asedb[i][structure.R] for i in range(len(asedb))]
    with ThreadPoolExecutor(2) as pool:
        got = list(pool.map(lambda i: asedb[i][structure.R], range(len(asedb))))
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_asedb_reads_in_dataloader_workers(asedb, context):
    """A forked worker opens its own connection; a spawned one gets none pickled."""
    expected = [asedb[i][structure.R] for i in range(len(asedb))]
    loader = torch.utils.data.DataLoader(
        asedb, batch_size=None, num_workers=2, multiprocessing_context=context
    )
    got = [item[structure.R] for item in loader]
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_atoms_loader_batches_from_workers_equal_the_main_process(asedb, context):
    """Batches from workers (sent as numpy arrays) are the same tensors, with the
    same dtypes, as batches collated in the main process."""
    from schnetpack.data import AtomsLoader

    expected = list(AtomsLoader(asedb, batch_size=2))
    got = list(
        AtomsLoader(asedb, batch_size=2, num_workers=2, multiprocessing_context=context)
    )
    assert len(got) == len(expected)
    for a, b in zip(expected, got):
        assert a.keys() == b.keys()
        for key in a:
            assert isinstance(b[key], torch.Tensor)
            assert b[key].dtype == a[key].dtype
            assert torch.equal(a[key], b[key])


def test_asedb_reads_rows_added_after_a_read(asedb, example_data):
    n = len(asedb)
    asedb[0]  # opens the held connection
    atoms, props = example_data[0]
    asedb.add_system(atoms=atoms, **props)
    assert torch.equal(asedb[n][structure.Z], torch.tensor(atoms.numbers))


def test_asedb_reads_scalar_properties(asedbpath, example_data):
    """A db written with ase.db directly may store a property as a plain number."""
    atoms, _ = example_data[0]
    with connect(asedbpath, use_lock_file=False) as conn:
        conn.metadata = {
            "_property_unit_dict": {"energy_per_atom": "eV"},
            "_distance_unit": "Ang",
            "atomrefs": {},
        }
        conn.write(atoms, data={"energy_per_atom": -1.5})
    props = ASEAtomsData(datapath=asedbpath)[0]
    assert props["energy_per_atom"].shape == (1,)
    assert float(props["energy_per_atom"]) == -1.5


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
            "property1": torch.tensor([1.0 + len(Z) * off]),
            "property2": torch.tensor([off]),
        }
        data.append(d)

    atomref = {"property1": torch.ones((100,)) / 3.0}
    for bs in range(1, 7):
        stats = calculate_stats(
            data,
            {"property1": True, "property2": False},
            atomref=atomref,
            batch_size=bs,
            num_workers=0,
        )
        assert np.allclose(stats["property1"][0].numpy(), np.array([0.0]))
        assert np.allclose(stats["property1"][1].numpy(), np.array([1.0]))
        assert np.allclose(stats["property2"][0].numpy(), np.array([0.0]))
        assert np.allclose(stats["property2"][1].numpy(), np.array([1.0]))


def test_asedb_add(asedb, example_data):
    l = len(asedb)

    at, props = example_data[0]
    asedb.add_system(atoms=at, **props)

    props = dict(props)
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
            assert isinstance(v, torch.Tensor), k
            assert (p2[k] == v).all(), v

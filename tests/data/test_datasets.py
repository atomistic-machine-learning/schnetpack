import json
import os

import numpy as np
import pytest
from ase.db import connect
from ase.units import GPa

from schnetpack.data import GroupSplit
from schnetpack.data.splitting import random_split
from schnetpack.datasets import MD17, QM7X, QM9, MPTraj, rMD17
from schnetpack.datasets.mptraj import iter_json_object
from schnetpack.datasets.qm7x import md5sum
from schnetpack.lightning import AtomsDataModule


@pytest.fixture
def test_qm9_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/test_qm9.db")
    return path


@pytest.mark.download
@pytest.mark.slow
def test_qm9(test_qm9_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = QM9(test_qm9_path, remove_uncharacterized=True)
    qm9 = AtomsDataModule(
        dataset,
        num_train=10,
        num_val=5,
        batch_size=5,
        split_file=str(tmp_path / "split.npz"),
    )
    qm9.setup()
    assert len(qm9.train_dataset) == 10
    assert len(qm9.val_dataset) == 5
    assert len(qm9.test_dataset) == 5

    ds = [b for b in qm9.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in qm9.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in qm9.test_dataloader()]
    assert len(ds) == 1


@pytest.fixture
def test_md17_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/tmp/test_md17.db")
    return path


@pytest.mark.download
@pytest.mark.slow
def test_md17(test_md17_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = MD17(test_md17_path, molecule="uracil")
    md17 = AtomsDataModule(
        dataset,
        num_train=10,
        num_val=5,
        num_test=5,
        batch_size=5,
        split_file=str(tmp_path / "split.npz"),
    )
    md17.setup()
    assert len(md17.train_dataset) == 10
    assert len(md17.val_dataset) == 5
    assert len(md17.test_dataset) == 5

    ds = [b for b in md17.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in md17.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in md17.test_dataloader()]
    assert len(ds) == 1


@pytest.fixture
def test_rmd17_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/tmp/test_rmd17.db")
    return path


@pytest.mark.download
@pytest.mark.slow
def test_rmd17(test_rmd17_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = rMD17(test_rmd17_path, molecule="uracil")
    md17 = AtomsDataModule(
        dataset,
        num_train=950,
        num_val=50,
        num_test=1000,
        batch_size=5,
        split_file=str(tmp_path / "split.npz"),
    )
    md17.setup()
    assert len(md17.train_dataset) == 950
    assert len(md17.val_dataset) == 50
    assert len(md17.test_dataset) == 1000

    train_idx = md17.train_dataset.subset_idx
    val_idx = md17.val_dataset.subset_idx
    test_idx = md17.test_dataset.subset_idx
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0


def _mptrj_frame(rng, elements, magmom=True):
    """One MPtrj frame: a pymatgen Structure dict plus VASP labels."""
    n = len(elements)
    cell = np.diag(rng.uniform(3.0, 5.0, 3)) + rng.uniform(-0.3, 0.3, (3, 3))
    stress = rng.normal(size=(3, 3))
    return {
        "structure": {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "lattice": {"matrix": cell.tolist()},
            "sites": [
                {
                    "species": [{"element": el, "occu": 1}],
                    "abc": rng.random(3).tolist(),
                    "properties": {},
                    "label": el,
                }
                for el in elements
            ],
        },
        "uncorrected_total_energy": float(rng.normal()),
        "corrected_total_energy": float(rng.normal()),
        "force": rng.normal(size=(n, 3)).tolist(),
        "stress": (stress + stress.T).tolist(),
        "magmom": rng.normal(size=n).tolist() if magmom else None,
        "bandgap": 0.0,
    }


@pytest.fixture
def mptrj_raw(tmp_path):
    """A small MPtrj JSON: material id -> frame id -> frame."""
    rng = np.random.default_rng(0)
    raw = {
        f"mp-{i}": {
            f"{i}-0-{j}": _mptrj_frame(rng, ["Fe", "O", "O"][: 1 + i % 3], j % 2 == 0)
            for j in range(1 + i % 4)
        }
        for i in range(8)
    }
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with open(raw_dir / MPTraj.json_file, "w") as f:
        json.dump(raw, f, indent=1)
    return raw, raw_dir


def test_iter_json_object_across_chunks(mptrj_raw):
    raw, raw_dir = mptrj_raw
    # chunks much smaller than one value: every value needs several refills
    items = list(iter_json_object(str(raw_dir / MPTraj.json_file), chunk_size=7))
    assert items == list(raw.items())


def test_mptraj(mptrj_raw, tmp_path, monkeypatch):
    raw, raw_dir = mptrj_raw
    monkeypatch.setattr(MPTraj, "json_md5", md5sum(str(raw_dir / MPTraj.json_file)))
    monkeypatch.setattr(MPTraj, "url", "http://unused")

    dataset = MPTraj(str(tmp_path / "mptraj.db"), raw_data_path=str(raw_dir))

    frames = [
        (mp_id, frame_id, frame)
        for mp_id, material in raw.items()
        for frame_id, frame in material.items()
    ]
    assert len(dataset) == len(frames)
    assert dataset.metadata["groups_ids"]["material_idx"] == [
        i for i, material in enumerate(raw.values()) for _ in material
    ]

    for idx, (mp_id, frame_id, frame) in enumerate(frames):
        row = dataset[idx]
        cell = np.array(frame["structure"]["lattice"]["matrix"])
        abc = np.array([site["abc"] for site in frame["structure"]["sites"]])
        np.testing.assert_allclose(row["_positions"].numpy(), abc @ cell)
        np.testing.assert_allclose(row["_cell"].numpy()[0], cell)
        assert row["_pbc"].all()
        assert row["energy"].item() == pytest.approx(frame["corrected_total_energy"])
        assert row["uncorrected_energy"].item() == pytest.approx(
            frame["uncorrected_total_energy"]
        )
        np.testing.assert_allclose(row["forces"].numpy(), frame["force"])
        # VASP kbar (positive under compression) -> eV/Ang^3 (ASE sign)
        np.testing.assert_allclose(
            row["stress"].numpy()[0], -0.1 * GPa * np.array(frame["stress"])
        )
        if frame["magmom"] is None:
            assert row["magmoms"].isnan().all()
        else:
            np.testing.assert_allclose(row["magmoms"].numpy(), frame["magmom"])

        kvp = connect(dataset.datapath).get(idx + 1).key_value_pairs
        assert kvp == {"material_id": mp_id, "frame_id": frame_id}

    # the frames of one material end up in one split
    train, val, test = GroupSplit("material_idx").split(dataset, 4, 2, None)
    groups = np.array(dataset.metadata["groups_ids"]["material_idx"])
    assert sorted(train + val + test) == list(range(len(dataset)))
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(groups[a]) & set(groups[b])


@pytest.mark.parametrize("sizes", [(50, 10, None), (15, 10, None), (0.8, 0.3, None)])
def test_random_split_refuses_more_than_the_dataset(sizes):
    """Sizes that need more structures than the dataset holds raise, instead of
    cutting the later partitions short."""
    with pytest.raises(ValueError, match="dataset holds 20"):
        random_split(20, *sizes)


def test_random_split_fills_the_rest():
    """The undefined partition gets the rest, also when that is nothing."""
    assert [len(p) for p in random_split(20, 15, 5, None)] == [15, 5, 0]
    assert [len(p) for p in random_split(20, 0.5, 0.25, None)] == [10, 5, 5]


def test_qm7x_property_names_are_distinct():
    """Each property needs its own name. Otherwise the property tables collapse and
    one property silently loads the data of another."""
    names = [
        QM7X.forces,
        QM7X.energy,
        QM7X.Eat,
        QM7X.EPBE0,
        QM7X.EMBD,
        QM7X.FPBE0,
        QM7X.FMBD,
        QM7X.RMSD,
    ]
    assert len(set(names)) == len(names)
    assert set(QM7X._native_property_units()) == set(names)
    assert set(QM7X.property_dataset_keys) == set(names)
    assert QM7X.property_dataset_keys[QM7X.FPBE0] == "pbe0FOR"
    assert QM7X.property_dataset_keys[QM7X.FMBD] == "vdwFOR"


@pytest.mark.parametrize(
    "name",
    [
        "ASEAtomsData",
        "AtomsCache",
        "AtomsLoader",
        "RandomSplit",
        "calculate_stats",
    ],
)
def test_datasets_still_pass_on_the_data_api(name):
    """Older releases exported all of schnetpack.data from schnetpack.datasets."""
    import schnetpack as spk

    assert getattr(spk.datasets, name) is getattr(spk.data, name)


def test_datasets_still_pass_on_the_deprecated_datamodule():
    import schnetpack as spk
    from schnetpack.lightning import AtomsDataModule

    with pytest.warns(DeprecationWarning, match="moved to"):
        assert spk.datasets.AtomsDataModule is AtomsDataModule


def test_qm7x_interrupted_extraction_leaves_no_partial_file(tmp_path, monkeypatch):
    """A partial file under the target name would be reused by later runs."""
    import lzma

    import schnetpack.datasets.qm7x as qm7x_module

    content = bytes(range(256)) * 4096
    source = tmp_path / "1000.xz"
    with lzma.open(source, "wb") as f:
        f.write(content)
    target = tmp_path / "1000.hdf5"

    def interrupted_copy(fin, fout):
        fout.write(fin.read(1000))
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(qm7x_module.shutil, "copyfileobj", interrupted_copy)
        with pytest.raises(KeyboardInterrupt):
            qm7x_module.extract_xz(str(source), str(target))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1000.xz"]

    qm7x_module.extract_xz(str(source), str(target))
    assert target.read_bytes() == content

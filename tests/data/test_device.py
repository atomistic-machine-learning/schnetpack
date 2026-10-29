import os

import numpy as np
import pytest
import torch
from ase import Atoms

import schnetpack.properties as properties
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.data.device import (
    DeviceAtomsData,
    OrderedBatches,
    ShuffledBatches,
    convert_to_arrays,
)
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.generative import (
    VE,
    Diffuse,
    EpsParametrization,
    GaussianInterpolant,
    GaussianPrior,
    Variable,
)
from schnetpack.lightning import DeviceAtomsDataModule

CUTOFF = 5.0


@pytest.fixture
def molecules_db(tmp_path):
    db = ASEAtomsData.create(
        os.path.join(tmp_path, "molecules.db"),
        distance_unit="Ang",
        property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
    )
    rng = np.random.RandomState(0)
    atoms, props = [], []
    for i in range(30):
        n = 2 + i % 7
        atoms.append(Atoms(numbers=rng.randint(1, 10, n), positions=rng.randn(n, 3)))
        props.append({"energy": np.array([rng.randn()]), "forces": rng.randn(n, 3)})
    db.add_systems(props, atoms)
    return db.datapath


@pytest.fixture
def dataset(molecules_db):
    return ASEAtomsData(molecules_db, load_properties=["energy", "forces"])


def collated(dataset, sel):
    """The library pipeline's batch of the same structures, all pairs."""
    samples = []
    for i in sel:
        s = dataset[int(i)]
        n = int(s[properties.n_atoms])
        a, b = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        keep = a != b
        s[properties.idx_i], s[properties.idx_j] = a[keep], b[keep]
        s[properties.offsets] = torch.zeros(int(keep.sum()), 3, dtype=torch.float64)
        samples.append(s)
    return _atoms_collate_fn(samples)


def test_padded_batch_matches_the_collated_batch(dataset, tmp_path):
    convert_to_arrays(dataset, os.path.join(tmp_path, "arrays"))
    data = DeviceAtomsData(os.path.join(tmp_path, "arrays"), "cpu")
    assert len(data) == len(dataset)
    sel = torch.tensor([3, 0, 7, 5, 12])
    ref = collated(dataset, sel)
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    batch = data.padded_batch(sel, N + 4, P + 10, CUTOFF)

    B = len(sel)
    assert torch.equal(batch[properties.n_atoms][:B], ref[properties.n_atoms])
    assert int(batch[properties.n_atoms][B]) == 4  # the padding structure
    for key in (properties.Z, properties.idx_m):
        assert torch.equal(batch[key][:N], ref[key])
    for key in (properties.R, "forces"):
        assert batch[key].dtype == ref[key].dtype
        torch.testing.assert_close(batch[key][:N], ref[key])
    torch.testing.assert_close(batch["energy"][:B], ref["energy"])
    # the same pairs, in the same order
    assert torch.equal(batch[properties.idx_i][:P], ref[properties.idx_i])
    assert torch.equal(batch[properties.idx_j][:P], ref[properties.idx_j])
    # padding: atoms at the origin in their own structure, pairs beyond the cutoff
    assert torch.equal(batch[properties.atom_mask], (torch.arange(N + 4) < N).double())
    assert torch.equal(
        batch[properties.structure_mask], (torch.arange(B + 1) < B).double()
    )
    assert (batch[properties.idx_m][N:] == B).all()
    assert (batch[properties.R][N:] == 0).all() and (batch["forces"][N:] == 0).all()
    assert (batch[properties.idx_i][P:] == N).all() and (
        batch[properties.idx_j][P:] == N
    ).all()
    assert (batch[properties.offsets][P:, 0] > CUTOFF).all()
    assert (batch[properties.offsets][:P] == 0).all()


def test_index_batches(dataset, tmp_path):
    convert_to_arrays(dataset, os.path.join(tmp_path, "arrays"))
    data = DeviceAtomsData(os.path.join(tmp_path, "arrays"), "cpu")
    idx = np.arange(25)
    pad = data.pad_sizes(idx, 4, k=6.0)
    shuffled = ShuffledBatches(data, idx, 4, pad, seed=1)
    batches = list(shuffled)
    assert len(batches) + shuffled.skipped == len(shuffled) == 6
    assert all(len(sel) == 4 and data.fits(sel, *pad) for sel in batches)
    assert not torch.equal(torch.cat(batches), torch.cat(list(shuffled)))  # new order
    ordered = list(OrderedBatches(idx, 4))
    assert len(ordered) == 7 and torch.equal(torch.cat(ordered), torch.from_numpy(idx))
    exact = data.exact_pad_sizes(idx, 4)
    assert all(data.fits(sel, *exact) for sel in ordered)


def positions():
    return Variable(
        name="positions",
        view=properties.R,
        per="atom",
        noising=GaussianInterpolant(
            VE(sigma_min=0.01, sigma_max=2.0), prior=GaussianPrior(std=2.0)
        ),
        parametrization=EpsParametrization(),
        output_key="eps",
    )


def test_datamodule_batches(dataset, tmp_path):
    split = os.path.join(tmp_path, "split.npz")
    np.savez(
        split, train_idx=np.arange(20), val_idx=np.arange(20, 27), test_idx=[27, 28]
    )
    v = positions()
    dm = DeviceAtomsDataModule(
        dataset,
        batch_size=4,
        split_file=split,
        cutoff=CUTOFF,
        transforms=[trn.SubtractCenterOfGeometry(), Diffuse([v]), trn.CastTo32()],
        val_batch_size=3,
        device="cpu",
        cache_dir=os.path.join(tmp_path, "arrays"),
    )
    dm.setup("fit")
    train = list(dm.train_dataloader())
    assert len(train) == 5
    shapes = {tuple(b[properties.R].shape) for b in train}
    assert len(shapes) == 1  # fixed shapes
    b = train[0]
    assert b[properties.R].dtype == torch.float32
    # one time per structure; weight 0 on padding atoms
    t, idx_m, mask = b[properties.t], b[properties.idx_m], b[properties.atom_mask]
    for m in range(5):
        assert t[idx_m == m].unique().numel() == 1
    assert (b[v.weight_key][mask == 0] == 0).all()
    assert (b[v.weight_key][mask == 1] == 1).all()

    # validation: the same noise in every pass, and the training draws go on
    first, second = list(dm.val_dataloader()), list(dm.val_dataloader())
    assert len(first) == 3
    for a, c in zip(first, second):
        assert torch.equal(a[properties.R], c[properties.R])
    assert not torch.equal(
        list(dm.train_dataloader())[0][properties.R], b[properties.R]
    )
    assert len(list(dm.test_dataloader())) == 1

    # the cache is reused, and statistics come from the train split
    mtime = os.path.getmtime(os.path.join(tmp_path, "arrays", "meta.json"))
    dm2 = DeviceAtomsDataModule(
        dataset,
        4,
        split,
        CUTOFF,
        device="cpu",
        cache_dir=os.path.join(tmp_path, "arrays"),
    )
    dm2.setup("fit")
    assert os.path.getmtime(os.path.join(tmp_path, "arrays", "meta.json")) == mtime
    mean, std = dm2.get_stats("energy", divide_by_atoms=False, remove_atomref=False)
    energies = torch.cat([dataset[i]["energy"] for i in range(20)])
    torch.testing.assert_close(mean, energies.mean())

import os

import numpy as np
import pytest
import torch
from ase import Atoms

import schnetpack.properties as properties
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.data.device import (
    ArrayStats,
    DeviceAtomsData,
    DeviceAtomsLoader,
    OrderedBatches,
    ShuffledBatches,
    convert_to_arrays,
    split_transforms,
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
from schnetpack.lightning import AtomsDataModule

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
    batch = data.padded_batch(sel, N + 4, P + 10, CUTOFF + 1.0)

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


def test_molecules_with_a_stored_neighbor_list(dataset, tmp_path):
    """The conversion's neighbor list (pairs within its cutoff) is what a padded
    batch of molecules holds, in its order."""
    cutoff = 1.5  # the test molecules are ~1-3 A across: some pairs in, some out
    neighbors = trn.MatScipyNeighborList(cutoff=cutoff)
    convert_to_arrays(
        dataset, os.path.join(tmp_path, "arrays"), neighbor_list=neighbors
    )
    data = DeviceAtomsData(os.path.join(tmp_path, "arrays"), "cpu")
    sel = torch.tensor([3, 0, 7, 5, 12])
    ref = _atoms_collate_fn([neighbors(dataset[int(i)]) for i in sel])
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    all_pairs = int((data.n_atoms_cpu[sel] * (data.n_atoms_cpu[sel] - 1)).sum())
    assert 0 < P < all_pairs
    assert int(data.n_pairs_cpu[sel].sum()) == P
    batch = data.padded_batch(sel, N + 1, P + 3, cutoff + 1.0)
    assert torch.equal(batch[properties.idx_i][:P], ref[properties.idx_i])
    assert torch.equal(batch[properties.idx_j][:P], ref[properties.idx_j])
    assert (batch[properties.idx_i][P:] == N).all()
    assert (batch[properties.offsets][P:, 0] > cutoff).all()


@pytest.mark.parametrize("divide_by_atoms", [False, True])
def test_array_statistics_match_the_db_pass(dataset, tmp_path, divide_by_atoms):
    """Mean, std and estimated atom references from the arrays equal those of
    calculate_stats / estimate_atomrefs over the same structures."""
    from schnetpack.data.stats import calculate_stats, estimate_atomrefs

    convert_to_arrays(dataset, os.path.join(tmp_path, "arrays"))
    data = DeviceAtomsData(os.path.join(tmp_path, "arrays"), "cpu")
    idx = [int(i) for i in np.random.RandomState(1).permutation(len(dataset))[:20]]
    atomref = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
    for ref in (None, atomref):
        mean, std = data.stats("energy", idx, divide_by_atoms, ref)
        expected = calculate_stats(
            dataset,
            {"energy": divide_by_atoms},
            atomref=None if ref is None else {"energy": ref},
            indices=idx,
        )["energy"]
        torch.testing.assert_close(mean, expected[0].double())
        torch.testing.assert_close(std, expected[1].double())
    got = data.atomrefs("energy", idx, is_extensive=not divide_by_atoms)
    expected = estimate_atomrefs(dataset, {"energy": not divide_by_atoms}, indices=idx)[
        "energy"
    ]
    torch.testing.assert_close(got.float(), expected, rtol=1e-4, atol=1e-4)


def test_budget_batches(dataset, tmp_path):
    """Batches packed to an atom and pair budget: fixed slots, free ones on the
    empty structure (masked out), each structure at most once per pass, the same
    number of batches every pass."""
    from schnetpack.data.device import BudgetBatches

    convert_to_arrays(dataset, os.path.join(tmp_path, "arrays"))
    data = DeviceAtomsData(os.path.join(tmp_path, "arrays"), "cpu")
    assert len(data) == len(dataset) and data.empty == len(dataset)
    idx = np.arange(28)
    budget = BudgetBatches(data, idx, batch_size=4, seed=0, grains=(1, 1))
    for _ in range(3):
        batches = list(budget)
        assert len(batches) == len(budget)
        seen = torch.cat([b[b != data.empty] for b in batches])
        assert len(seen) == len(set(seen.tolist()))
        for sel in batches:
            assert len(sel) == budget.max_structures
            real = sel[sel != data.empty]
            assert int(data.n_atoms_cpu[real].sum()) <= budget.atom_budget
            assert int(data.n_pairs_cpu[real].sum()) <= budget.pair_budget
            batch = data.padded_batch(sel, *budget.pad, CUTOFF)
            mask = batch[properties.structure_mask]
            assert torch.equal(mask[:-1], (sel != data.empty).double())
            assert int(batch[properties.atom_mask].sum()) == int(
                data.n_atoms_cpu[real].sum()
            )
            assert (batch["energy"][:-1][sel == data.empty] == 0).all()


def test_remove_offsets_with_atomrefs_on_a_batch(dataset):
    """RemoveOffsets subtracts each structure's own atom references on a collated
    batch, as it does structure by structure."""
    atomref = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
    t = trn.RemoveOffsets("energy", remove_mean=False, remove_atomrefs=True)
    t.atomref = atomref
    sel = torch.tensor([3, 0, 7])
    one_by_one = torch.cat([t(dataset[int(i)])["energy"] for i in sel])
    batch = t(collated(dataset, sel))
    torch.testing.assert_close(batch["energy"], one_by_one)


def test_periodic_structures_with_a_stored_neighbor_list(tmp_path):
    """Crystals: the conversion's neighbor list (periodic images) is stored as local
    pairs and cell shifts; a padded batch holds the same pairs, offsets, cells and
    pbc flags as the collated per-structure neighbor lists."""
    from ase.build import bulk

    path = os.path.join(tmp_path, "crystals.db")
    db = ASEAtomsData.create(
        path, distance_unit="Ang", property_unit_dict={"energy": "eV"}
    )
    rng = np.random.RandomState(0)
    atoms, props = [], []
    for k in range(6):
        a = bulk("Si", "diamond", a=5.43).repeat((1, 1, 1 + k % 2))
        a.rattle(0.05, seed=k)
        atoms.append(a)
        props.append({"energy": np.array([rng.randn()])})
    db.add_systems(props, atoms)
    dataset = ASEAtomsData(path, load_properties=["energy"])
    neighbors = trn.MatScipyNeighborList(cutoff=3.0)
    arrays = os.path.join(tmp_path, "arrays")
    convert_to_arrays(dataset, arrays, neighbor_list=neighbors)
    data = DeviceAtomsData(arrays, "cpu")
    sel = torch.tensor([2, 0, 5])
    ref = _atoms_collate_fn([neighbors(dataset[int(i)]) for i in sel])
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    assert int(data.n_pairs_cpu[sel].sum()) == P
    batch = data.padded_batch(sel, N + 2, P + 5, 4.0)
    assert torch.equal(batch[properties.idx_i][:P], ref[properties.idx_i])
    assert torch.equal(batch[properties.idx_j][:P], ref[properties.idx_j])
    torch.testing.assert_close(batch[properties.offsets][:P], ref[properties.offsets])
    assert (batch[properties.offsets][:P] != 0).any()  # periodic images present
    torch.testing.assert_close(batch[properties.cell][:3], ref[properties.cell])
    assert torch.equal(batch[properties.pbc][:9], ref[properties.pbc])
    assert (batch[properties.offsets][P:, 0] > 3.0).all()


def test_periodic_structures_need_a_neighbor_list(tmp_path):
    from ase.build import bulk

    path = os.path.join(tmp_path, "crystals.db")
    db = ASEAtomsData.create(path, distance_unit="Ang", property_unit_dict={})
    db.add_systems([{}], [bulk("Si", "diamond", a=5.43)])
    with pytest.raises(ValueError, match="neighbor_list"):
        convert_to_arrays(ASEAtomsData(path), os.path.join(tmp_path, "arrays"))


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


def test_from_dataset_converts_once(dataset, tmp_path):
    """The arrays are written once and reused while dataset and neighbor list stay
    the same; another neighbor list converts again. Padding pairs lie 1 beyond the
    stored pairs' cutoff, or far out with all pairs."""
    arrays = os.path.join(tmp_path, "arrays")
    data = DeviceAtomsData.from_dataset(dataset, "cpu", array_dir=arrays)
    assert data.pair_i is None and data.pad_distance == 1000.0
    mtime = os.path.getmtime(os.path.join(arrays, "meta.json"))
    DeviceAtomsData.from_dataset(dataset, "cpu", array_dir=arrays)
    assert os.path.getmtime(os.path.join(arrays, "meta.json")) == mtime
    neighbors = trn.MatScipyNeighborList(cutoff=1.5)
    data = DeviceAtomsData.from_dataset(dataset, "cpu", neighbors, arrays)
    assert data.pair_i is not None and data.pad_distance == 2.5
    batch = data.padded_batch(torch.tensor([0, 1]), 32, 64)
    assert (
        batch[properties.offsets][int(data.n_pairs_cpu[:2].sum()) :, 0] == 2.5
    ).all()


def test_split_transforms():
    neighbors, cast = trn.MatScipyNeighborList(cutoff=2.0), trn.CastTo32()
    nl, rest = split_transforms([neighbors, cast])
    assert nl is neighbors and rest == [cast]
    nl, rest = split_transforms([trn.AllToAllNeighborList(), cast])
    assert nl is None and rest == [cast]
    with pytest.raises(ValueError, match="one neighbor list"):
        split_transforms([neighbors, trn.ASENeighborList(cutoff=2.0)])


def test_array_stats_answer_as_the_db_provider(dataset, tmp_path):
    from schnetpack.data import StatsAtomrefProvider

    data = DeviceAtomsData.from_dataset(
        dataset, "cpu", array_dir=os.path.join(tmp_path, "arrays")
    )
    idx = list(range(20))
    stats, db = (
        ArrayStats(data, idx, dataset.atomrefs),
        StatsAtomrefProvider(dataset, idx),
    )
    for divide_by_atoms in (False, True):
        for remove_atomref in (False, True):
            for a, b in zip(
                stats.get_stats("energy", divide_by_atoms, remove_atomref),
                db.get_stats("energy", divide_by_atoms, remove_atomref),
            ):
                torch.testing.assert_close(a, b.double())
        torch.testing.assert_close(
            stats.get_atomrefs("energy", divide_by_atoms)["energy"].float(),
            db.get_atomrefs("energy", divide_by_atoms)["energy"].float(),
            rtol=1e-4,
            atol=1e-4,
        )


def test_device_atoms_loader(dataset, tmp_path):
    """Batches built on the device with the transforms applied: in order every
    structure once, shuffled and budget batches with one shape each; index_batches
    with build_batch give the same batches."""
    data = DeviceAtomsData.from_dataset(
        dataset, "cpu", array_dir=os.path.join(tmp_path, "arrays")
    )
    idx = np.arange(25)
    ordered = DeviceAtomsLoader(data, idx, 4, [trn.CastTo32()])
    batches = list(ordered)
    assert len(batches) == len(ordered) == 7
    assert batches[0][properties.R].dtype == torch.float32
    real = [b[properties.idx][:-1] for b in batches]  # the last is the padding
    assert torch.equal(torch.cat(real), torch.from_numpy(idx))
    for batching in ("fixed", "budget"):
        loader = DeviceAtomsLoader(data, idx, 4, batching=batching, seed=3)
        shapes = {tuple(b[properties.R].shape) for b in loader}
        assert len(shapes) == 1
        for sel in loader.index_batches:
            batch = loader.build_batch(sel)
            assert batch[properties.R].shape == (loader.pad[0], 3)
    with pytest.raises(ValueError, match="batching"):
        DeviceAtomsLoader(data, idx, 4, batching="sorted")


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


def test_datamodule_batches(molecules_db, dataset, tmp_path):
    """AtomsDataModule with data_device: fixed-shape batches built on the device,
    the dataset's transforms run on them, seeded validation noise, a reused cache
    and statistics from the arrays."""
    split = os.path.join(tmp_path, "split.npz")
    np.savez(
        split, train_idx=np.arange(20), val_idx=np.arange(20, 27), test_idx=[27, 28]
    )
    v = positions()
    noised = ASEAtomsData(
        molecules_db,
        load_properties=["energy", "forces"],
        transforms=[trn.SubtractCenterOfGeometry(), Diffuse([v]), trn.CastTo32()],
    )
    dm = AtomsDataModule(
        noised,
        batch_size=4,
        num_train=20,
        num_val=7,
        split_file=split,
        val_batch_size=3,
        data_device="cpu",
        array_dir=os.path.join(tmp_path, "arrays"),
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
    dm2 = AtomsDataModule(
        dataset,
        4,
        20,
        7,
        split_file=split,
        data_device="cpu",
        array_dir=os.path.join(tmp_path, "arrays"),
    )
    dm2.setup("fit")
    assert os.path.getmtime(os.path.join(tmp_path, "arrays", "meta.json")) == mtime
    mean, std = dm2.get_stats("energy", divide_by_atoms=False, remove_atomref=False)
    energies = torch.cat([dataset[i]["energy"] for i in range(20)])
    torch.testing.assert_close(mean, energies.mean())


def test_datamodule_takes_the_neighbor_list_to_the_conversion(dataset, tmp_path):
    """A neighbor list among the dataset's transforms runs at the conversion; the
    batches hold its pairs, and budget batching packs them."""
    split = os.path.join(tmp_path, "split.npz")
    np.savez(split, train_idx=np.arange(24), val_idx=np.arange(24, 30), test_idx=[])
    neighbors = ASEAtomsData(
        dataset.datapath,
        load_properties=["energy"],
        transforms=[trn.MatScipyNeighborList(cutoff=1.5), trn.CastTo32()],
    )
    dm = AtomsDataModule(
        neighbors,
        batch_size=4,
        num_train=24,
        num_val=6,
        split_file=split,
        data_device="cpu",
        batching="budget",
        array_dir=os.path.join(tmp_path, "arrays"),
    )
    dm.setup("fit")
    assert dm.data.pair_i is not None
    batch = next(iter(dm.train_dataloader()))
    real = batch[properties.idx_i][batch[properties.offsets][:, 0] == 0]
    d = (
        batch[properties.R][batch[properties.idx_j]]
        - batch[properties.R][batch[properties.idx_i]]
    ).norm(dim=1)
    assert len(real) > 0 and (d[batch[properties.offsets][:, 0] == 0] < 1.5).all()
    budget = dm.device_loaders["train"].index_batches
    assert len(batch[properties.structure_mask]) == budget.max_structures + 1


def test_batching_budget_needs_data_device(dataset):
    with pytest.raises(ValueError, match="data_device"):
        AtomsDataModule(dataset, 4, 20, 7, batching="budget")

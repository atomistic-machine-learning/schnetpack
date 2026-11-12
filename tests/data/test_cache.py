import multiprocessing
import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.transform as trn
from schnetpack.data import (
    ArrayStats,
    ASEAtomsData,
    AtomsCache,
    AtomsLoader,
    BudgetBatches,
    OrderedBatches,
    StatsAtomrefProvider,
    calculate_stats,
    estimate_atomrefs,
    fit_atomrefs,
    structure_stats,
)
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.generative import (
    VE,
    Diffuse,
    EpsParametrization,
    GaussianInterpolant,
    GaussianPrior,
    PeriodicGaussianInterpolant,
    PeriodicPositions,
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


def crystals_db(tmp_path, n, energies=False):
    path = os.path.join(tmp_path, "crystals.db")
    units = {"energy": "eV"} if energies else {}
    db = ASEAtomsData.create(path, distance_unit="Ang", property_unit_dict=units)
    atoms = []
    for k in range(n):
        a = bulk("Si", "diamond", a=5.43).repeat((1, 1, 1 + k % 2))
        a.rattle(0.05, seed=k)
        atoms.append(a)
    rng = np.random.RandomState(0)
    props = [{"energy": np.array([rng.randn()])} if energies else {} for _ in atoms]
    db.add_systems(props, atoms)
    return path


def cache_of(path, transforms=(), **kwargs) -> AtomsCache:
    """The structures of a db with some transforms, read once into a cache."""
    return AtomsCache.from_dataset(
        ASEAtomsData(path, transforms=list(transforms), **kwargs)
    )


def flat(source, key, indices):
    """What structure_stats and fit_atomrefs take: y, n_atoms, owner, Z."""
    f = source.read_properties([key], indices)
    return f[key], f[properties.n_atoms], f[properties.idx_m], f[properties.Z]


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


def noised_positions():
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


def periodic_positions():
    return Variable(
        name="positions",
        view=PeriodicPositions(),
        per="atom",
        noising=PeriodicGaussianInterpolant(VE(sigma_min=0.01, sigma_max=1.0)),
        parametrization=EpsParametrization(),
        output_key="eps",
    )


def real_rows(batch):
    """A padded batch without its padding: the real structures, atoms and pairs."""
    S = int(batch[properties.structure_mask].sum())
    N = int(batch[properties.atom_mask].sum())
    P = int((batch[properties.idx_i] < N).sum()) if properties.idx_i in batch else 0
    out = {}
    for key, value in batch.items():
        if key in (properties.atom_mask, properties.structure_mask):
            continue
        if key in (properties.idx_i, properties.idx_j, properties.offsets):
            out[key] = value[:P]
        elif key == properties.pbc:
            out[key] = value[: 3 * S]
        elif value.shape[0] == batch[properties.n_atoms].shape[0]:
            out[key] = value[:S]
        else:
            out[key] = value[:N]
    return out


# ----------------------------------------------------------------- the cache's batches


def test_padded_batch_matches_the_collated_batch(dataset, molecules_db):
    cache = cache_of(molecules_db, [trn.AllToAllNeighborList()])
    assert len(cache) == len(dataset)
    sel = torch.tensor([3, 0, 7, 5, 12])
    ref = collated(dataset, sel)
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    batch = cache.gather(sel, N + 4, P + 10)

    B = len(sel)
    assert torch.equal(batch[properties.n_atoms][:B], ref[properties.n_atoms])
    assert int(batch[properties.n_atoms][B]) == 4  # the padding structure
    assert torch.equal(batch[properties.idx][:B], ref[properties.idx])  # db index
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
    assert (batch[properties.idx_i][P:] == N).all()
    assert (batch[properties.idx_j][P:] == N).all()
    assert (batch[properties.offsets][P:, 0] > CUTOFF).all()
    assert (batch[properties.offsets][:P] == 0).all()


def test_molecules_with_a_stored_neighbor_list(dataset, molecules_db):
    """The pairs of the dataset's neighbor list (pairs within its cutoff) are what a
    padded batch of molecules holds, in its order."""
    cutoff = 1.5  # the test molecules are ~1-3 A across: some pairs in, some out
    neighbors = trn.MatScipyNeighborList(cutoff=cutoff)
    cache = cache_of(molecules_db, [neighbors])
    sel = torch.tensor([3, 0, 7, 5, 12])
    ref = _atoms_collate_fn([neighbors(dataset[int(i)]) for i in sel])
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    all_pairs = int((cache.n_atoms[sel] * (cache.n_atoms[sel] - 1)).sum())
    assert 0 < P < all_pairs
    assert int(cache.n_pairs[sel].sum()) == P
    batch = cache.gather(sel, N + 1, P + 3)
    assert torch.equal(batch[properties.idx_i][:P], ref[properties.idx_i])
    assert torch.equal(batch[properties.idx_j][:P], ref[properties.idx_j])
    assert (batch[properties.idx_i][P:] == N).all()
    assert (batch[properties.offsets][P:, 0] > cutoff).all()


def test_periodic_structures_with_a_stored_neighbor_list(tmp_path):
    """Crystals: the pairs of the neighbor list (periodic images) are kept as local
    pairs and cell shifts; a padded batch holds the same pairs, offsets, cells and
    pbc flags as the collated per-structure neighbor lists."""
    path = crystals_db(tmp_path, 6, energies=True)
    dataset = ASEAtomsData(path, load_properties=["energy"])
    neighbors = trn.MatScipyNeighborList(cutoff=3.0)
    cache = cache_of(path, [neighbors], load_properties=["energy"])
    sel = torch.tensor([2, 0, 5])
    ref = _atoms_collate_fn([neighbors(dataset[int(i)]) for i in sel])
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    assert int(cache.n_pairs[sel].sum()) == P
    batch = cache.gather(sel, N + 2, P + 5)
    assert torch.equal(batch[properties.idx_i][:P], ref[properties.idx_i])
    assert torch.equal(batch[properties.idx_j][:P], ref[properties.idx_j])
    torch.testing.assert_close(batch[properties.offsets][:P], ref[properties.offsets])
    assert (batch[properties.offsets][:P] != 0).any()  # periodic images present
    torch.testing.assert_close(batch[properties.cell][:3], ref[properties.cell])
    assert torch.equal(batch[properties.pbc][:9], ref[properties.pbc])
    assert (batch[properties.offsets][P:, 0] > 3.0).all()


def test_stress_on_padded_periodic_batches(tmp_path):
    """Energy, forces and stress of the crystals in a padded batch equal those of the
    collated batch: the padding atoms, pairs and structure (identity cell) add
    nothing to the strain derivative of a real structure."""
    path = crystals_db(tmp_path, 4, energies=True)
    dataset = ASEAtomsData(path, load_properties=["energy"])
    neighbors = trn.MatScipyNeighborList(cutoff=3.0)
    cache = cache_of(path, [neighbors], load_properties=["energy"])
    sel = torch.tensor([1, 0, 3])
    ref = _atoms_collate_fn([neighbors(dataset[int(i)]) for i in sel])
    N, P = len(ref[properties.Z]), len(ref[properties.idx_i])
    padded = cache.gather(sel, N + 3, P + 7)

    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.PaiNN(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=3.0),
            cutoff_fn=spk.nn.CosineCutoff(3.0),
        ),
        input_modules=[spk.model.Strain(), spk.model.PairwiseDistances()],
        output_modules=[
            spk.model.Atomwise(n_in=16, output_key="energy"),
            spk.model.Forces(calc_stress=True),
        ],
    ).double()
    expected, got = model(ref), model(padded)
    assert expected["stress"].abs().max() > 1e-6
    torch.testing.assert_close(got["energy"][:3], expected["energy"])
    torch.testing.assert_close(got["forces"][:N], expected["forces"])
    torch.testing.assert_close(got["stress"][:3], expected["stress"])
    assert torch.isfinite(got["stress"]).all()  # the padding structure has a volume
    assert (got["forces"][N:] == 0).all()


# ------------------------------------------------------------------------ statistics


@pytest.mark.parametrize("divide_by_atoms", [False, True])
def test_cache_statistics_match_the_db_pass(dataset, molecules_db, divide_by_atoms):
    """Mean, std and estimated atom references from the cache equal those of
    calculate_stats / estimate_atomrefs over the same structures."""
    cache = cache_of(molecules_db, load_properties=["energy", "forces"])
    idx = [int(i) for i in np.random.RandomState(1).permutation(len(dataset))[:20]]
    atomref = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
    for ref in (None, atomref):
        mean, std = structure_stats(*flat(cache, "energy", idx), divide_by_atoms, ref)
        expected = calculate_stats(
            dataset,
            {"energy": divide_by_atoms},
            atomref=None if ref is None else {"energy": ref},
            indices=idx,
        )["energy"]
        torch.testing.assert_close(mean, expected[0].double())
        torch.testing.assert_close(std, expected[1].double())
    got = fit_atomrefs(*flat(cache, "energy", idx), is_extensive=not divide_by_atoms)
    expected = estimate_atomrefs(dataset, {"energy": not divide_by_atoms}, indices=idx)
    torch.testing.assert_close(
        got.float(), expected["energy"].float(), rtol=1e-4, atol=1e-4
    )


def test_array_stats_equal_the_db_provider(dataset, molecules_db):
    cache = cache_of(molecules_db, load_properties=["energy", "forces"])
    idx = list(range(20))
    stats, db = ArrayStats(cache, idx), StatsAtomrefProvider(dataset, idx)
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


# -------------------------------------------------------------------------- samplers


def test_budget_batches(molecules_db):
    """Batches packed to an atom and pair budget, from the counts alone: each
    structure at most once per pass, the same number of batches every pass."""
    cache = cache_of(molecules_db, [trn.AllToAllNeighborList()])
    atoms, pairs = cache.n_atoms.numpy(), cache.n_pairs.numpy()
    assert np.array_equal(pairs, atoms * (atoms - 1))
    budget = BudgetBatches(atoms, pairs, np.arange(28), 4, seed=0, grains=(1, 1))
    for _ in range(3):
        batches = list(budget)
        assert len(batches) == len(budget)
        seen = np.concatenate(batches)
        assert len(seen) == len(set(seen.tolist()))
        for sel in batches:
            assert 0 < len(sel) <= budget.max_structures
            assert int(atoms[sel].sum()) <= budget.atom_budget
            assert int(pairs[sel].sum()) <= budget.pair_budget


def test_budget_batches_of_fewer_structures_than_a_batch():
    """A split smaller than one batch gives that one batch in every pass."""
    n_atoms, n_pairs = np.full(5, 3), np.full(5, 6)
    batches = BudgetBatches(n_atoms, n_pairs, np.arange(5), batch_size=100)
    for _ in range(2):
        (batch,) = list(batches)
        assert sorted(batch.tolist()) == [0, 1, 2, 3, 4]


def test_ordered_batches(dataset, molecules_db):
    """Every structure once, in order, the last batch smaller; the ordered loader
    pads to its largest batch."""
    cache = cache_of(molecules_db, [trn.AllToAllNeighborList()])
    idx = np.arange(25)
    ordered = list(OrderedBatches(idx, 4))
    assert len(ordered) == 7 and np.array_equal(np.concatenate(ordered), idx)
    loader = AtomsLoader(dataset.subset(list(idx)), 4, cache=cache)
    batches = list(loader)
    atoms = [int(cache.n_atoms[sel].sum()) for sel in ordered]
    pairs = [int(cache.n_pairs[sel].sum()) for sel in ordered]
    for b in batches:
        assert b[properties.R].shape[0] == max(atoms) + 1
        assert b[properties.idx_i].shape[0] == max(pairs)
    real = [b[properties.idx][:-1] for b in batches]  # the last is the padding
    assert torch.equal(torch.cat(real), torch.from_numpy(idx))


def test_shuffled_batches_have_fixed_slots(dataset, molecules_db):
    """A shuffled loader's batches: one shape, one number of structure slots, the
    free slots empty (their index is -1), which the batch masks out."""
    cache = cache_of(molecules_db, load_properties=["energy", "forces"])
    loader = AtomsLoader(dataset.subset(list(range(28))), 4, shuffle=True, cache=cache)
    shapes = set()
    for batch in loader:
        sel = batch[properties.idx][:-1]  # the last entry is the padding structure
        assert len(sel) == loader.batch_sampler.max_structures
        real = sel[sel >= 0]
        mask = batch[properties.structure_mask]
        assert torch.equal(mask[:-1], (sel >= 0).double())
        assert int(batch[properties.atom_mask].sum()) == int(cache.n_atoms[real].sum())
        assert (batch["energy"][:-1][sel < 0] == 0).all()
        shapes.add(tuple(batch[properties.R].shape))
    assert len(shapes) == 1


# ------------------------------------------------------------------- the two routes


@pytest.mark.parametrize(
    "neighbors", [trn.MatScipyNeighborList(cutoff=1.5), trn.AllToAllNeighborList()]
)
def test_the_two_routes_give_equal_real_rows(molecules_db, neighbors):
    """One pair of lists serves both routes: per structure on every read, or once
    into the cache; the real rows of the padded batches equal the collated ones."""
    data = ASEAtomsData(
        molecules_db,
        load_properties=["energy", "forces"],
        transforms=[trn.SubtractCenterOfMass(), neighbors],
    )
    train_idx, val_idx = list(range(20)), list(range(20, 30))
    on_batches = [
        trn.RemoveOffsets(
            "energy", remove_mean=True, remove_atomrefs=True, estimate_atomref=True
        )
    ]
    cache = AtomsCache.from_dataset(data)
    for t in on_batches:
        t.initialize(ArrayStats(cache, train_idx, data.atomrefs))
    view = data.subset(val_idx)
    default = AtomsLoader(view, batch_size=4, batch_transforms=on_batches)
    cached = AtomsLoader(view, batch_size=4, cache=cache, batch_transforms=on_batches)
    pairs = 0
    for a, b in zip(default, cached, strict=True):
        b = real_rows(b)
        for key in b:
            torch.testing.assert_close(b[key], a[key], msg=key)
        pairs += len(a[properties.idx_i])
    assert pairs > 0


def test_a_cache_refuses_transforms_that_move_atoms_or_need_statistics(molecules_db):
    noise = Diffuse([noised_positions()])
    assert noise.moves_atoms
    with pytest.raises(ValueError, match="moves the atoms"):
        cache_of(molecules_db, [trn.SubtractCenterOfGeometry(), noise])
    with pytest.raises(ValueError, match="needs statistics"):
        cache_of(molecules_db, [trn.RemoveOffsets("energy", remove_mean=True)])
    # noise on atom types only leaves the geometry: a neighbor list may come first
    types = Variable(
        name="types",
        view=properties.Z,
        per="atom",
        noising=GaussianInterpolant(
            VE(sigma_min=0.01, sigma_max=2.0), prior=GaussianPrior(std=2.0)
        ),
        parametrization=EpsParametrization(),
        output_key="eps_Z",
    )
    assert not Diffuse([types]).moves_atoms


def test_positions_of_a_view(dataset, molecules_db):
    """A view's structures are found in the cache by their db index; a structure the
    cache does not hold raises."""
    part = ASEAtomsData(molecules_db, subset_idx=[4, 9, 2, 17])
    cache = AtomsCache.from_dataset(part)
    assert cache.index.tolist() == [4, 9, 2, 17]
    assert cache.positions(part.subset([2, 0])).tolist() == [2, 0]
    assert cache.positions(dataset.subset([17, 4])).tolist() == [3, 0]
    with pytest.raises(ValueError, match=r"does not hold the structures \[5\]"):
        cache.positions(dataset.subset([4, 5]))
    with pytest.raises(ValueError, match="does not hold"):
        cache.positions(dataset)  # all 30


def test_a_cached_loader_refuses_the_per_structure_arguments(dataset, molecules_db):
    cache = cache_of(molecules_db)
    for kwargs in (
        dict(num_workers=2),
        dict(sampler=[0, 1]),
        dict(batch_sampler=[[0, 1]], batch_size=1),
        dict(collate_fn=lambda x: x),
        dict(pin_memory=True),
    ):
        name = next(iter(kwargs))
        with pytest.raises(ValueError, match=f"takes no {name}"):
            AtomsLoader(dataset, cache=cache, **kwargs)


@pytest.mark.parametrize("use_cache", [False, True])
def test_transform_seed_repeats_the_draws(molecules_db, use_cache):
    """Validation noise: the same draws in every pass, on both routes; without a
    seed new ones."""
    data = ASEAtomsData(
        molecules_db, load_properties=[], transforms=[trn.SubtractCenterOfGeometry()]
    )
    cache = AtomsCache.from_dataset(data) if use_cache else None
    noise = [Diffuse([noised_positions()])]
    view = data.subset(list(range(10)))
    seeded = AtomsLoader(view, 4, cache=cache, batch_transforms=noise, transform_seed=0)
    first, second = list(seeded), list(seeded)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a[properties.R], b[properties.R])
    free = AtomsLoader(view, 4, cache=cache, batch_transforms=noise)
    assert not torch.equal(next(iter(free))[properties.R], first[0][properties.R])


def test_a_cached_loader_leaves_the_global_generator_untouched(dataset, molecules_db):
    cache = cache_of(molecules_db)
    loader = AtomsLoader(dataset.subset(list(range(20))), 4, shuffle=True, cache=cache)
    state = torch.random.get_rng_state()
    batches = list(loader)
    assert batches and torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("use_cache", [False, True])
def test_a_loader_moves_the_batches_and_transforms_them(
    dataset, molecules_db, use_cache
):
    cache = cache_of(molecules_db) if use_cache else None
    loader = AtomsLoader(
        dataset.subset([3, 1, 4]),
        2,
        cache=cache,
        batch_transforms=[trn.CastTo32()],
        device="meta",
    )
    batch = next(iter(loader))
    assert batch[properties.R].device.type == "meta"
    assert batch[properties.R].dtype == torch.float32


def test_a_loader_survives_lightnings_rebuild(dataset):
    """Lightning rebuilds a DataLoader subclass from its attributes to inject a
    sampler (DDP): the batch transforms, device and seeds come along."""
    from pytorch_lightning.utilities.data import _update_dataloader
    from torch.utils.data import SequentialSampler

    view = dataset.subset([5, 2, 11, 7])
    loader = AtomsLoader(
        view,
        2,
        shuffle=True,
        num_workers=0,
        batch_transforms=[trn.CastTo32()],
        device="cpu",
        transform_seed=3,
    )
    rebuilt = _update_dataloader(loader, SequentialSampler(view))
    assert type(rebuilt) is AtomsLoader
    assert rebuilt.batch_transforms == loader.batch_transforms
    assert rebuilt.device == loader.device and rebuilt.transform_seed == 3
    batches = list(rebuilt)
    assert torch.cat([b[properties.idx] for b in batches]).tolist() == [5, 2, 11, 7]
    assert batches[0][properties.R].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batches_on_a_gpu_equal_the_batches_in_host_memory(dataset, molecules_db):
    """For a GPU the batches are gathered into pinned tensors that are reused, and
    copied without waiting: every batch still holds its own structures, also after
    the following batches were built."""
    cache = cache_of(
        molecules_db,
        [trn.MatScipyNeighborList(cutoff=CUTOFF)],
        load_properties=["energy", "forces"],
    )
    view = dataset.subset(list(range(25)))
    on_gpu, in_host = (
        AtomsLoader(
            view,
            4,
            shuffle=True,
            seed=3,
            cache=cache,
            batch_transforms=[trn.CastTo32()],
            device=d,
        )
        for d in ("cuda", "cpu")
    )
    batches = list(on_gpu)  # all of them first: the pinned tensors are rewritten
    assert len(batches) == len(on_gpu) > 1
    for a, b in zip(batches, in_host):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].device.type == "cuda" and a[key].dtype == b[key].dtype
            assert torch.equal(a[key].cpu(), b[key]), key


# --------------------------------------------------------------------------- crystals


def test_periodic_noise_on_padded_batches(tmp_path):
    """The padding and the empty structure have an identity cell, so noise that
    wraps positions into their cells runs on padded batches: real atoms end up
    inside their cell, padding rows stay finite and get weight 0."""
    from schnetpack.utils.cell import to_frac

    path = crystals_db(tmp_path, 6)
    data = ASEAtomsData(
        path, load_properties=[], transforms=[trn.MatScipyNeighborList(cutoff=3.0)]
    )
    v = periodic_positions()
    cache = AtomsCache.from_dataset(data)
    loader = AtomsLoader(
        data, 2, shuffle=True, cache=cache, batch_transforms=[Diffuse([v])]
    )
    batches = list(loader)
    assert batches
    for batch in batches:
        real = batch[properties.atom_mask] == 1
        cells = batch[properties.cell][batch[properties.idx_m]]
        frac = to_frac(batch[properties.R], cells)
        assert ((frac[real] >= 0) & (frac[real] < 1 + 1e-9)).all()
        assert torch.isfinite(batch[properties.R]).all()
        assert torch.isfinite(batch[v.label_key]).all()
        assert (batch[v.weight_key][~real] == 0).all()
        assert torch.equal(batch[properties.cell][-1], torch.eye(3, dtype=cells.dtype))


def test_neighbor_list_on_noised_crystals(tmp_path):
    """Pairs built on every batch, after the noise: the pairs matscipy finds in the
    noised crystals, padded to one capacity, so the batches keep one shape."""
    path = crystals_db(tmp_path, 8)
    cutoff = 3.0
    on_batches = trn.BatchCutoffNeighborList(cutoff)
    data = ASEAtomsData(path, load_properties=[])
    cache = AtomsCache.from_dataset(data)
    assert cache.pair_i is None  # no pairs stored
    loader = AtomsLoader(
        data,
        2,
        shuffle=True,
        cache=cache,
        batch_transforms=[Diffuse([periodic_positions()]), on_batches],
    )
    batches = [b for _ in range(2) for b in loader]
    assert len({tuple(b[properties.idx_i].shape) for b in batches}) == 1
    rows = batches[0][properties.R].shape[0]
    assert batches[0][properties.idx_i].shape[0] == on_batches.pair_capacity[rows]
    reference = trn.MatScipyNeighborList(cutoff=cutoff)
    for batch in batches:
        n_real = int(batch[properties.atom_mask].sum())
        real = batch[properties.idx_i] < n_real
        samples, start = [], 0
        for k, n in enumerate(batch[properties.n_atoms][:-1].tolist()):
            if n == 0:
                continue
            samples.append(
                reference(
                    {
                        properties.idx: torch.tensor([k]),
                        properties.Z: batch[properties.Z][start : start + n],
                        properties.R: batch[properties.R][start : start + n],
                        properties.cell: batch[properties.cell][k][None],
                        properties.pbc: torch.tensor([True, True, True]),
                        properties.n_atoms: torch.tensor([n]),
                    }
                )
            )
            start += n
        ref = _atoms_collate_fn(samples)

        def pairs(i, j, offsets, R):
            d = (R[j] - R[i] + offsets).numpy().round(6)
            return sorted(zip(i.tolist(), j.tolist(), map(tuple, d.tolist())))

        got = pairs(
            batch[properties.idx_i][real],
            batch[properties.idx_j][real],
            batch[properties.offsets][real],
            batch[properties.R],
        )
        want = pairs(
            ref[properties.idx_i],
            ref[properties.idx_j],
            ref[properties.offsets],
            ref[properties.R],
        )
        assert got == want and len(got) > 0
        # padding pairs: the first padding atom with itself, beyond the cutoff
        assert (batch[properties.idx_i][~real] == n_real).all()
        assert (batch[properties.idx_j][~real] == n_real).all()
        assert (batch[properties.offsets][~real][:, 0] == cutoff + 1.0).all()
    # without any neighbor list the batches hold no pairs
    assert properties.idx_i not in next(iter(AtomsLoader(data, 2, cache=cache)))


def test_joint_crystal_noise_on_padded_batches(tmp_path):
    """Cell, atom types and positions noised together on padded batches, with the
    neighbor list rebuilt after the noise: every entry finite, weight 0 on padding
    rows, and the padding and empty structures keep their identity cell."""
    from schnetpack.generative import (
        CellParameters,
        FlowMatching,
        LogitsParametrization,
        MaskNoising,
        VPCosine,
    )

    path = crystals_db(tmp_path, 8)
    cell = Variable(
        name="cell",
        view=CellParameters(length=2.94, angle=99.7, var_length=2.52, var_angle=297.0),
        per="structure",
        noising=GaussianInterpolant(
            VPCosine(kappa=0.008, t_min=1e-3),
            prior=GaussianPrior(
                std=1.0, centered=False, key="cell_y", per="structure", dim=6
            ),
        ),
        parametrization=EpsParametrization(),
        output_key="eps_cell",
    )
    types = Variable(
        name="types",
        view=properties.Z,
        per="atom",
        noising=MaskNoising(FlowMatching(t_min=1e-3, t_max=1.0)),
        parametrization=LogitsParametrization(n_classes=101),
        output_key="type_logits",
    )
    positions = periodic_positions()
    on_batches = trn.BatchCutoffNeighborList(cutoff=3.0)
    data = ASEAtomsData(path, load_properties=[])
    cache = AtomsCache.from_dataset(data)
    transforms = [Diffuse([cell, types, positions]), on_batches, trn.CastTo32()]
    loader = AtomsLoader(
        data, 2, shuffle=True, cache=cache, batch_transforms=transforms
    )
    batches = list(loader)
    assert batches
    eye = torch.eye(3)
    for batch in batches:
        for key, value in batch.items():
            if value.is_floating_point():
                assert torch.isfinite(value).all(), key
        atoms = batch[properties.atom_mask]
        structures = batch[properties.structure_mask]
        for v, mask in ((cell, structures), (types, atoms), (positions, atoms)):
            assert (batch[v.weight_key][mask == 0] == 0).all()
            assert (batch[v.weight_key][mask == 1] > 0).any()
        cells = batch[properties.cell]
        assert torch.equal(
            cells[structures == 0], eye.expand_as(cells[structures == 0])
        )
        assert not torch.equal(cells[structures == 1][0], eye)
        rows = batch[properties.R].shape[0]
        assert batch[properties.idx_i].shape[0] == on_batches.pair_capacity[rows]


# ------------------------------------------------------------------- saving, datamodule


def test_cache_saved_and_loaded(molecules_db, tmp_path):
    """from_dataset writes nothing; save writes a folder (through .tmp) that load
    reads back equal; a folder of other files is refused."""
    before = sorted(os.listdir(tmp_path))
    cache = cache_of(molecules_db, [trn.MatScipyNeighborList(cutoff=1.5)])
    assert sorted(os.listdir(tmp_path)) == before  # nothing written
    assert properties.lidx_i in cache.arrays
    path = str(tmp_path / "pairs.cache")
    cache.save(path)
    cache.save(path)  # replaces the earlier save
    assert not os.path.exists(path + ".tmp")
    loaded = AtomsCache.load(path)
    assert loaded.meta == cache.meta and loaded.arrays.keys() == cache.arrays.keys()
    for key, values in cache.arrays.items():
        assert torch.equal(loaded.arrays[key], values), key
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="other files"):
        cache.save(str(other))
    assert os.listdir(other) == ["notes.txt"] and not os.path.exists(f"{other}.tmp")


def test_cache_stands_without_the_db(tmp_path):
    """Units and atom references are kept with the cache; the structures of a
    subset are kept with their db index."""
    path = os.path.join(tmp_path, "refs.db")
    refs = [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, -38.0]
    db = ASEAtomsData.create(
        path, "Ang", {"energy": "eV"}, atomrefs={"energy": refs + [0.0] * 93}
    )
    atoms = [
        Atoms("CH4", positions=np.random.RandomState(k).randn(5, 3)) for k in range(6)
    ]
    db.add_systems([{"energy": np.array([float(k)])} for k in range(6)], atoms)
    dataset = ASEAtomsData(
        path, property_units={"energy": "kcal/mol"}, subset_idx=[4, 1, 3]
    )
    folder = os.path.join(tmp_path, "cache")
    AtomsCache.from_dataset(dataset).save(folder)
    os.remove(path)
    alone = AtomsCache.load(folder)
    assert len(alone) == 3 and alone.units == {"energy": "kcal/mol"}
    assert alone.distance_unit == "Ang" and alone.index.tolist() == [4, 1, 3]
    kcal = 23.060548
    torch.testing.assert_close(
        alone.read_properties(["energy"])["energy"],
        torch.tensor([4.0, 1.0, 3.0], dtype=torch.float64) * kcal,
        rtol=1e-6,
        atol=0.0,
    )
    assert float(alone.atomrefs["energy"][1]) == pytest.approx(-0.5 * kcal, rel=1e-5)


def _make_cache(datapath, folder):
    data = ASEAtomsData(datapath, transforms=[trn.MatScipyNeighborList(cutoff=1.5)])
    AtomsDataModule(
        data, 4, 20, 5, split_file=None, cache=True, cache_dir=folder
    )._make_cache()


def test_two_processes_make_the_cache_once(molecules_db, tmp_path):
    """Two processes (DDP ranks) asking for the same cache_dir: one makes and saves
    the cache under the file lock, the other waits and loads it."""
    folder = str(tmp_path / "shared.cache")
    ctx = multiprocessing.get_context("fork")
    workers = [
        ctx.Process(target=_make_cache, args=(molecules_db, folder)) for _ in range(2)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=120)
    assert [w.exitcode for w in workers] == [0, 0]
    assert not os.path.exists(folder + ".tmp")
    assert len(AtomsCache.load(folder)) == 30


@pytest.mark.parametrize("use_cache", [False, True])
def test_datamodule_batches(molecules_db, tmp_path, use_cache):
    """AtomsDataModule on both routes: the batch transforms run on every batch,
    validation noise is the same in every pass, and with a cache the batches have
    one shape, the cache is reused and statistics come from it."""
    split = os.path.join(tmp_path, "split.npz")
    np.savez(
        split, train_idx=np.arange(20), val_idx=np.arange(20, 27), test_idx=[27, 28]
    )
    v = noised_positions()
    data = ASEAtomsData(
        molecules_db,
        load_properties=["energy", "forces"],
        transforms=[trn.SubtractCenterOfGeometry()],
    )
    folder = os.path.join(tmp_path, "cache")
    dm = AtomsDataModule(
        data,
        batch_size=4,
        num_train=20,
        num_val=7,
        split_file=split,
        val_batch_size=3,
        batch_transforms=[Diffuse([v]), trn.CastTo32()],
        cache=use_cache,
        cache_dir=folder,
    )
    dm.setup("fit")
    train = list(dm.train_dataloader())
    assert len(train) == len(dm.train_dataloader()) >= 3
    b = train[0]
    assert b[properties.R].dtype == torch.float32
    # one time per structure, weight 1 on real atoms
    t, idx_m = b[properties.t], b[properties.idx_m]
    real = b.get(properties.atom_mask, torch.ones_like(t)) == 1
    for m in idx_m[real].unique():
        assert t[idx_m == m].unique().numel() == 1
    assert (b[v.weight_key][real] == 1).all()
    if use_cache:
        assert len({tuple(b[properties.R].shape) for b in train}) == 1
        assert (b[v.weight_key][~real] == 0).all()

    # validation: the same noise in every pass, and the training draws go on
    first, second = list(dm.val_dataloader()), list(dm.val_dataloader())
    assert len(first) == 3
    for a, c in zip(first, second):
        assert torch.equal(a[properties.R], c[properties.R])
    assert len(list(dm.test_dataloader())) == 1
    assert os.path.exists(os.path.join(folder, "meta.json")) == use_cache
    if not use_cache:
        return

    # the cache is reused, and statistics come from the train split
    mtime = os.path.getmtime(os.path.join(folder, "meta.json"))
    dm2 = AtomsDataModule(
        data, 4, 20, 7, split_file=split, cache=True, cache_dir=folder
    )
    dm2.setup("fit")
    assert os.path.getmtime(os.path.join(folder, "meta.json")) == mtime
    mean, _ = dm2.get_stats("energy", divide_by_atoms=False, remove_atomref=False)
    energies = torch.cat([data[i]["energy"] for i in range(20)])
    torch.testing.assert_close(mean, energies.mean())

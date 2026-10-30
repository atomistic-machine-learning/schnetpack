"""
A dataset held on a device (a GPU), with batches built there.

:func:`convert_to_arrays` reads an :class:`~schnetpack.data.ASEAtomsData` once into
flat arrays, one file per key with the structures back to back, in the dataset's
units and dtypes. :class:`DeviceAtomsData` holds them on a device and builds padded
batches of any structures with index operations only: fixed shapes and no host
synchronization, so a training step can build its own batch inside a CUDA graph
(:class:`~schnetpack.lightning.AtomisticTask` with ``cuda_graph``). On QM9 this takes
the data loading out of the step: there are no per-sample reads or collation, and no
worker processes.

The pairs of a batch come from a neighbor list given to the conversion (e.g.
``MatScipyNeighborList(cutoff)``), which runs once per structure: each pair is stored
as its two local atoms and its cell shift (offset = shift @ cell), so crystals keep
their periodic images. Stored pairs suit fixed geometries (energies, forces). Without
a neighbor list, a batch holds every ordered atom pair (i, j), i != j, within each
structure (molecules): the neighbor list of noised structures in generative training,
whose positions change every time a structure is drawn.

Padding atoms (Z = 0, at the origin) form one extra structure, the last; padding pairs
join its first atom with itself through an offset beyond any cutoff, so the cutoff
function zeroes their messages and gradients. ``properties.atom_mask`` and
``properties.structure_mask`` mark the real entries (1.0) and the padding (0.0).
"""

import json
import os

import numpy as np
import torch

from schnetpack import properties
from schnetpack.data.atoms import ASEAtomsData

__all__ = [
    "convert_to_arrays",
    "DeviceAtomsData",
    "ShuffledBatches",
    "BudgetBatches",
    "OrderedBatches",
]

#: version of the files convert_to_arrays writes (2: cells, pbc, stored pairs)
ARRAYS_FORMAT = 2

_STRUCTURE_KEYS = (
    properties.idx,
    properties.Z,
    properties.R,
    properties.cell,
    properties.pbc,
    properties.n_atoms,
)


def _as_numpy(samples: list[dict[str, torch.Tensor]]) -> list[dict[str, np.ndarray]]:
    """Worker results as numpy arrays: pickled by value. Tensors would be passed
    through shared memory, one file descriptor each, and a batch of samples can
    exceed the open-file limit (1024 on Hydra), which hangs the loader."""
    return [{k: v.numpy() for k, v in s.items()} for s in samples]


def neighbor_list_descriptor(neighbor_list) -> dict | None:
    """What stored pairs depend on: the neighbor list's class and cutoff."""
    if neighbor_list is None:
        return None
    return {
        "type": type(neighbor_list).__name__,
        "cutoff": float(neighbor_list._cutoff),
    }


def _cell_shifts(offsets: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """The integer cell shifts S of pair offsets (offset = S @ cell)."""
    if not offsets.any():
        return np.zeros(offsets.shape, dtype=np.int8)
    shifts = np.rint(np.linalg.solve(cell.T, offsets.T).T)
    if np.abs(shifts @ cell - offsets).max() > 1e-6 * max(np.abs(cell).max(), 1.0):
        raise ValueError("pair offsets are not integer multiples of the cell vectors")
    if np.abs(shifts).max() > np.iinfo(np.int8).max:
        raise ValueError("a pair spans more than 127 cells")
    return shifts.astype(np.int8)


def convert_to_arrays(
    dataset: ASEAtomsData,
    out_dir: str,
    num_workers: int = 0,
    batch_size: int = 1000,
    neighbor_list=None,
) -> dict:
    """
    Read every structure of a dataset into flat arrays (written to out_dir).

    The dataset is read without its transforms, in index order, so array
    position k holds dataset index k. Per-atom properties (first dimension
    n_atoms) are concatenated; per-structure ones (first dimension 1) are
    stacked. Cells and pbc flags are kept per structure.

    Args:
        dataset: the dataset
        out_dir: directory for the ``.npy`` files and ``meta.json``
        num_workers: DataLoader workers reading the db
        batch_size: structures per read
        neighbor_list: a neighbor list transform (e.g.
            ``MatScipyNeighborList(cutoff)``), run on every structure in the
            workers: its pairs are stored as local atoms and integer cell shifts.
            Required for periodic structures.

    Returns:
        the metadata written to ``meta.json``

    Raises:
        ValueError: for periodic structures without a neighbor list
    """
    raw = dataset.subset(list(range(len(dataset))))
    raw.transforms = [] if neighbor_list is None else [neighbor_list]
    raw.split = None
    loader = torch.utils.data.DataLoader(
        raw,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_as_numpy,
        timeout=600 if num_workers else 0,  # raise rather than hang
    )
    columns: dict[str, list[np.ndarray]] = {}
    kinds: dict[str, str] = {}
    pairs: dict[str, list[np.ndarray]] = {"pair_i": [], "pair_j": [], "shift": []}
    n_pairs: list[int] = []
    pair_keys = (properties.idx_i, properties.idx_j, properties.offsets)
    for samples in loader:
        for s in samples:
            cell = s[properties.cell].reshape(3, 3)
            if neighbor_list is None and bool(s[properties.pbc].any()):
                raise ValueError(
                    "periodic structures need a neighbor_list (periodic images)"
                )
            if neighbor_list is not None:
                pairs["pair_i"].append(s[properties.idx_i])
                pairs["pair_j"].append(s[properties.idx_j])
                pairs["shift"].append(_cell_shifts(s[properties.offsets], cell))
                n_pairs.append(len(s[properties.idx_i]))
            columns.setdefault("cell", []).append(cell[None])
            columns.setdefault("pbc", []).append(s[properties.pbc].reshape(1, 3))
            n = int(s[properties.n_atoms].reshape(-1)[0])
            for key, value in s.items():
                if key in (properties.idx, properties.cell, properties.pbc):
                    continue
                if key in pair_keys:
                    continue  # the neighbor list's pairs, stored above
                v = value.reshape(-1) if value.ndim == 0 else value
                if key not in (properties.Z, properties.R, properties.n_atoms):
                    if v.shape[0] == n and n != 1:
                        kinds[key] = "atom"
                    elif v.shape[0] == 1:
                        kinds.setdefault(key, "structure")
                    else:
                        raise ValueError(
                            f"property {key!r} has shape {v.shape}: neither per "
                            f"atom ({n} atoms) nor per structure (1)"
                        )
                columns.setdefault(key, []).append(v)
    arrays = {
        "n_atoms": np.concatenate(columns.pop(properties.n_atoms)).astype(np.int64),
        "Z": np.concatenate(columns.pop(properties.Z)).astype(np.int64),
        "R": np.concatenate(columns.pop(properties.R)),
        "cell": np.concatenate(columns.pop("cell")),
        "pbc": np.concatenate(columns.pop("pbc")).astype(bool),
    }
    if neighbor_list is not None:
        small = int(arrays["n_atoms"].max()) <= np.iinfo(np.int16).max
        index = np.int16 if small else np.int32
        arrays["pair_i"] = np.concatenate(pairs["pair_i"]).astype(index)
        arrays["pair_j"] = np.concatenate(pairs["pair_j"]).astype(index)
        arrays["shift"] = np.concatenate(pairs["shift"]).reshape(-1, 3)
        arrays["n_pairs"] = np.asarray(n_pairs, dtype=np.int64)
    for key, values in columns.items():
        arrays[key] = np.concatenate(values)  # (1, ...) rows stack to (S, ...)
    os.makedirs(out_dir, exist_ok=True)
    names = {}
    for k, (key, a) in enumerate(arrays.items()):
        names[key] = f"{k}.npy"  # batch keys may hold characters unfit for files
        np.save(os.path.join(out_dir, names[key]), a)
    meta = {
        "structures": int(len(arrays["n_atoms"])),
        "atoms": int(len(arrays["Z"])),
        "files": names,
        "properties": {key: kinds[key] for key in columns},
        "source": dataset_fingerprint(dataset),
        "neighbor_list": neighbor_list_descriptor(neighbor_list),
        "format": ARRAYS_FORMAT,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def dataset_fingerprint(dataset: ASEAtomsData) -> dict:
    """What a converted copy depends on: the db file and how it is read."""
    stat = os.stat(dataset.datapath)
    return {
        "datapath": os.path.abspath(dataset.datapath),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "length": len(dataset),
        "subset_idx": None
        if dataset.subset_idx is None
        else [int(i) for i in dataset.subset_idx],
        "load_properties": list(dataset.load_properties),
        "conversions": {k: float(v) for k, v in dataset.conversions.items()},
        "distance_conversion": float(dataset.distance_conversion),
    }


class DeviceAtomsData:
    """
    The arrays of :func:`convert_to_arrays` on a device, and padded batches of any
    of their structures (module docstring).
    """

    def __init__(self, path: str, device: str | torch.device = "cuda"):
        """
        Args:
            path: directory written by :func:`convert_to_arrays`
            device: where the arrays and the batches live
        """
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        files = self.meta["files"]

        def load(key):
            return torch.from_numpy(np.load(os.path.join(path, files[key])))

        self.device = torch.device(device)
        self.n_atoms_cpu = load("n_atoms")
        self.n_atoms = self.n_atoms_cpu.to(self.device)
        self.atom_start = torch.cumsum(self.n_atoms, 0) - self.n_atoms
        self.Z = load("Z").to(self.device)
        self.R = load("R").to(self.device)
        self.properties = {
            key: (load(key).to(self.device), kind)
            for key, kind in self.meta["properties"].items()
        }
        n_structures = len(self.n_atoms_cpu)
        self.cell = (
            load("cell").to(self.device, self.R.dtype)
            if "cell" in files
            else torch.zeros(n_structures, 3, 3, dtype=self.R.dtype, device=self.device)
        )
        self.pbc = (
            load("pbc").to(self.device)
            if "pbc" in files
            else torch.zeros(n_structures, 3, dtype=torch.bool, device=self.device)
        )
        self.pair_start = self.pair_i = self.pair_j = self.shift = None
        if "pair_i" in files:  # stored by the conversion's neighbor list
            self.pair_i = load("pair_i").to(self.device)
            self.pair_j = load("pair_j").to(self.device)
            self.shift = load("shift").to(self.device)
            self.n_pairs_cpu = load("n_pairs")
            counts = self.n_pairs_cpu.to(self.device)
            self.pair_start = torch.cumsum(counts, 0) - counts
        else:  # every ordered pair of each structure
            self.n_pairs_cpu = self.n_atoms_cpu * (self.n_atoms_cpu - 1)
        # one empty structure (no atoms, no pairs, zero properties) after the real
        # ones: the free slots of a budget batch (BudgetBatches) point to it
        self.empty = len(self.n_atoms_cpu)
        none = torch.zeros(1, dtype=self.n_atoms_cpu.dtype)
        self.n_atoms_cpu = torch.cat([self.n_atoms_cpu, none])
        self.n_pairs_cpu = torch.cat([self.n_pairs_cpu, none])
        self.n_atoms = self.n_atoms_cpu.to(self.device)
        self.n_pairs = self.n_pairs_cpu.to(self.device)
        self.atom_start = torch.cat([self.atom_start, self.atom_start.new_zeros(1)])
        self.cell = torch.cat([self.cell, self.cell.new_zeros(1, 3, 3)])
        self.pbc = torch.cat([self.pbc, self.pbc.new_zeros(1, 3)])
        if self.pair_start is not None:
            self.pair_start = torch.cat([self.pair_start, self.pair_start.new_zeros(1)])
        for key, (v, kind) in self.properties.items():
            if kind == "structure":
                self.properties[key] = (
                    torch.cat([v, v.new_zeros(1, *v.shape[1:])]),
                    kind,
                )

    def __len__(self) -> int:
        return self.empty

    def _structures(self, key: str, indices) -> tuple[torch.Tensor, ...]:
        """A per-structure property of the structures indices (float64), their atom
        counts, and the structure (0..) and atomic number of each of their atoms."""
        values, kind = self.properties[key]
        if kind != "structure":
            raise ValueError(f"{key!r} is a per-atom property")
        idx = torch.as_tensor(indices, device=self.device)
        y = values[idx].double().reshape(len(idx))
        n = self.n_atoms[idx]
        owner = torch.repeat_interleave(torch.arange(len(idx), device=self.device), n)
        local = (
            torch.arange(len(owner), device=self.device)
            - (torch.cumsum(n, 0) - n)[owner]
        )
        return y, n, owner, self.Z[self.atom_start[idx][owner] + local]

    def stats(
        self,
        key: str,
        indices,
        divide_by_atoms: bool,
        atomref: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and standard deviation of a per-structure property over the structures
        indices, as :func:`~schnetpack.data.calculate_stats` computes them (float64,
        population standard deviation), from the arrays on the device.

        Args:
            key: the property
            indices: the structures, e.g. the training split
            divide_by_atoms: per atom, divided by the atom count
            atomref: reference value per atomic number, subtracted first
        """
        y, n, owner, Z = self._structures(key, indices)
        if atomref is not None:
            ref = atomref.to(self.device, torch.float64)[Z]
            y = y - torch.zeros_like(y).index_add_(0, owner, ref)
        if divide_by_atoms:
            y = y / n
        mean = y.mean()
        return mean.cpu(), ((y - mean) ** 2).mean().sqrt().cpu()

    def atomrefs(
        self, key: str, indices, is_extensive: bool, z_max: int = 100
    ) -> torch.Tensor:
        """
        Atom references of a property by least squares over the atom-type counts of
        the structures indices, as :func:`~schnetpack.data.estimate_atomrefs` fits
        them, in float64 (estimate_atomrefs works in float32 and per molecule).

        Returns:
            reference per atomic number, shape (z_max,); 0 for absent elements
        """
        y, n, owner, Z = self._structures(key, indices)
        counts = torch.zeros(len(y), z_max, dtype=torch.float64, device=self.device)
        counts.index_put_((owner, Z), torch.ones_like(owner, dtype=torch.float64), True)
        present = counts.sum(0) != 0
        X = counts[:, present]
        if not is_extensive:  # estimate_atomrefs: y * n / (atoms per structure)
            y = y * n / X.sum(1)
        out = torch.zeros(z_max, dtype=torch.float64, device=self.device)
        out[present] = torch.linalg.solve(X.T @ X, X.T @ y)
        return out.cpu()

    def fits(self, sel: torch.Tensor, n_atoms_pad: int, n_pairs_pad: int) -> bool:
        """Whether the structures sel (CPU) fit the padded sizes (one padding atom
        at least). CPU only."""
        return (
            int(self.n_atoms_cpu[sel].sum()) < n_atoms_pad
            and int(self.n_pairs_cpu[sel].sum()) <= n_pairs_pad
        )

    def pad_sizes(
        self, indices, batch_size: int, k: float = 6.0, grains=(64, 1024)
    ) -> tuple[int, int]:
        """
        Padded sizes for random batches of the structures indices: atoms and pairs
        of batch_size random structures, mean + k standard deviations (+1 padding
        atom), rounded up to the grains. A batch exceeds them with probability
        ~1e-9 at k = 6 (it is then skipped).
        """
        sizes = []
        for per, grain, extra in (
            (self.n_atoms_cpu, grains[0], 1),
            (self.n_pairs_cpu, grains[1], 0),
        ):
            x = per[torch.as_tensor(indices)].double()
            size = batch_size * x.mean() + k * x.std() * batch_size**0.5 + extra
            sizes.append(int(-(-float(size) // grain) * grain))
        return sizes[0], sizes[1]

    def exact_pad_sizes(self, indices, batch_size: int) -> tuple[int, int]:
        """Padded sizes that fit every batch of indices taken in order (largest
        batch, +1 padding atom)."""
        indices = torch.as_tensor(indices)
        atoms = self.n_atoms_cpu[indices]
        pairs = self.n_pairs_cpu[indices]
        starts = range(0, len(indices), batch_size)
        return (
            max(int(atoms[s : s + batch_size].sum()) for s in starts) + 1,
            max(int(pairs[s : s + batch_size].sum()) for s in starts),
        )

    def padded_batch(
        self,
        sel: torch.Tensor,
        n_atoms_pad: int,
        n_pairs_pad: int,
        pad_distance: float = 1000.0,
    ) -> dict[str, torch.Tensor]:
        """
        The batch of the structures sel (device tensor), padded to fixed sizes.

        Fixed-shape operations only (searchsorted instead of repeat_interleave):
        recordable in a CUDA graph with sel as its only input. The caller checks
        :meth:`fits` on the CPU.

        Args:
            sel: indices of the structures, shape (B,)
            n_atoms_pad: atoms of the batch, padding included
            n_pairs_pad: pairs of the batch, padding included
            pad_distance: length of the padding pairs, beyond the model's cutoff

        Returns:
            the batch, B + 1 structures (the last one is the padding)
        """
        dev, R_dtype = self.device, self.R.dtype
        B = sel.shape[0]
        n, p = self.n_atoms[sel], self.n_pairs[sel]
        cn, cp = torch.cumsum(n, 0), torch.cumsum(p, 0)
        first, pfirst = cn - n, cp - p
        # atom slots: their structure (B: the padding structure)
        k = torch.arange(n_atoms_pad, device=dev)
        b = torch.searchsorted(cn, k, right=True)
        real = b < B
        bc = b.clamp(max=B - 1)
        atom = torch.where(real, self.atom_start[sel][bc] + k - first[bc], 0)
        # pair slots: their structure, and the pair's local atoms
        k2 = torch.arange(n_pairs_pad, device=dev)
        b2 = torch.searchsorted(cp, k2, right=True)
        real2 = b2 < B
        b2c = b2.clamp(max=B - 1)
        q = k2 - pfirst[b2c]
        if self.pair_i is None:  # every ordered pair i != j
            nm1 = (n[b2c] - 1).clamp(min=1)
            i = torch.div(q, nm1, rounding_mode="floor")
            r = q - i * nm1
            j = r + (r >= i).long()
        else:  # the stored pairs of the conversion's neighbor list
            last = max(self.pair_i.shape[0] - 1, 0)
            g = (self.pair_start[sel][b2c] + q).clamp(0, last)
            i, j = self.pair_i[g].long(), self.pair_j[g].long()
        pad_atom = cn[-1:].expand(n_pairs_pad)  # the first padding atom
        offsets = torch.zeros(n_pairs_pad, 3, device=dev, dtype=R_dtype)
        offsets[:, 0] = torch.where(real2, 0.0, pad_distance)
        if self.shift is not None:  # periodic images: offset = shift @ cell
            cells = self.cell[sel][b2c]
            image = (self.shift[g].to(R_dtype)[:, :, None] * cells).sum(1)
            offsets = torch.where(real2[:, None], image, offsets)
        mask = real.to(R_dtype)
        batch = {
            properties.idx: torch.cat([sel, sel[:1]]),
            properties.Z: torch.where(real, self.Z[atom], 0),
            properties.R: self.R[atom] * mask[:, None],
            properties.idx_i: torch.where(real2, first[b2c] + i, pad_atom),
            properties.idx_j: torch.where(real2, first[b2c] + j, pad_atom),
            properties.offsets: offsets,
            properties.idx_m: torch.where(real, b, B),
            properties.n_atoms: torch.cat([n, n_atoms_pad - cn[-1:]]),
            properties.cell: torch.cat([self.cell[sel], self.cell.new_zeros(1, 3, 3)]),
            properties.pbc: torch.cat(
                [self.pbc[sel], self.pbc.new_zeros(1, 3)]
            ).reshape(-1),
            properties.atom_mask: mask,
            properties.structure_mask: torch.cat(  # free slots: empty structures
                [(n > 0).to(R_dtype), torch.zeros(1, device=dev, dtype=R_dtype)]
            ),
        }
        for key, (v, kind) in self.properties.items():
            if kind == "structure":
                pad = torch.zeros(1, *v.shape[1:], dtype=v.dtype, device=dev)
                batch[key] = torch.cat([v[sel], pad])
            else:
                batch[key] = v[atom] * mask.view(-1, *[1] * (v.dim() - 1)).to(v.dtype)
        return batch


class ShuffledBatches:
    """
    Index batches of a shuffled epoch (CPU tensors, pinned for a GPU): the
    structures of each training batch. The last incomplete batch is dropped and
    batches that do not fit the padded sizes are skipped (counted in ``skipped``),
    so every batch has the same shapes. Each pass draws a new order.
    """

    def __init__(self, data: DeviceAtomsData, indices, batch_size, pad, seed=0):
        self.data, self.indices = data, np.asarray(indices)
        self.batch_size, self.pad = batch_size, pad
        self.rng = np.random.default_rng(seed)
        self.skipped = 0

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size

    def __iter__(self):
        order = self.rng.permutation(self.indices)
        pin = self.data.device.type == "cuda"
        for s in range(0, len(order) - self.batch_size + 1, self.batch_size):
            sel = torch.from_numpy(order[s : s + self.batch_size])
            if not self.data.fits(sel, *self.pad):
                self.skipped += 1
                continue
            yield sel.pin_memory() if pin else sel


class BudgetBatches:
    """
    Index batches packed to an atom and a pair budget instead of a fixed number of
    structures. Structures are taken from a shuffled order while the batch's atoms
    and pairs fit the budgets, so a batch of large structures holds fewer of them,
    and the padding fills only what is left of the budget instead of covering the
    worst case of ``batch_size`` structures. The budgets are what ``batch_size``
    structures hold on average. Every batch has ``max_structures`` slots; the free
    ones point to the empty structure (``data.empty``, ``structure_mask`` 0), so the
    loss is the mean over the structures actually in the batch.

    Every pass yields the same number of batches, as a fixed epoch length needs: the
    full batches of one packing. Structures left over wait for the next pass's
    order; a pass that packs fewer batches is topped up from a second order.
    """

    def __init__(
        self,
        data: DeviceAtomsData,
        indices,
        batch_size: int,
        seed: int = 0,
        max_ratio: float = 1.3,
        grains=(32, 256),
    ):
        """
        Args:
            data: the arrays
            indices: the structures to draw from (e.g. the training split)
            batch_size: structures per batch on average
            seed: seed of the orders
            max_ratio: slots per batch, relative to batch_size
            grains: the budgets are rounded up to these (atoms, pairs)
        """
        self.data, self.indices = data, np.asarray(indices)
        self.rng = np.random.default_rng(seed)
        self.atoms = data.n_atoms_cpu.numpy()
        self.pairs = data.n_pairs_cpu.numpy()
        budgets = []
        for per, grain in ((self.atoms, grains[0]), (self.pairs, grains[1])):
            x = per[self.indices]
            size = max(batch_size * float(x.mean()), float(x.max()), 1.0)
            budgets.append(int(-(-size // grain) * grain))
        self.atom_budget, self.pair_budget = budgets
        self.max_structures = int(np.ceil(max_ratio * batch_size))
        self.pad = (self.atom_budget + 1, self.pair_budget)  # + one padding atom
        self.n_batches = max(len(self._pack(self.rng.permutation(self.indices))) - 1, 1)
        self.skipped = 0  # as ShuffledBatches; packed batches always fit

    def _pack(self, order) -> list[list[int]]:
        batches, current, atoms, pairs = [], [], 0, 0
        for s, a, p in zip(
            order.tolist(), self.atoms[order].tolist(), self.pairs[order].tolist()
        ):
            full = len(current) == self.max_structures
            if current and (
                full or atoms + a > self.atom_budget or pairs + p > self.pair_budget
            ):
                batches.append(current)
                current, atoms, pairs = [], 0, 0
            current.append(s)
            atoms, pairs = atoms + a, pairs + p
        batches.append(current)  # the last one, usually not full
        return batches

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        batches = self._pack(self.rng.permutation(self.indices))[:-1]
        while len(batches) < self.n_batches:
            batches += self._pack(self.rng.permutation(self.indices))[:-1]
        pin = self.data.device.type == "cuda"
        for b in batches[: self.n_batches]:
            sel = np.full(self.max_structures, self.data.empty, dtype=np.int64)
            sel[: len(b)] = b
            sel = torch.from_numpy(sel)
            yield sel.pin_memory() if pin else sel


class OrderedBatches:
    """Index batches of all structures in their order, the last one possibly smaller
    (validation and testing)."""

    def __init__(self, indices, batch_size):
        self.indices, self.batch_size = np.asarray(indices), batch_size

    def __len__(self) -> int:
        return -(-len(self.indices) // self.batch_size)

    def __iter__(self):
        for s in range(0, len(self.indices), self.batch_size):
            yield torch.from_numpy(self.indices[s : s + self.batch_size])

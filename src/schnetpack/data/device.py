"""
A dataset held on a device (a GPU), with batches built there.

:func:`convert_to_arrays` reads an :class:`~schnetpack.data.ASEAtomsData` once into
flat arrays, one file per key with the structures back to back, in the dataset's
units and dtypes. :class:`DeviceAtomsData` holds them on a device and builds padded
batches of any structures with index operations only: fixed shapes and no host
synchronization, so a training step can build its own batch inside a CUDA graph
(:class:`~schnetpack.lightning.GraphStepTask`). On QM9 this takes the data loading
out of the step: there are no per-sample reads or collation, and no worker processes.

A padded batch holds every ordered atom pair (i, j), i != j, within each structure:
the neighbor list of molecules without a cutoff, as used for noised structures in
generative training, whose positions change every time a structure is drawn. Padding
atoms (Z = 0, at the origin) form one extra structure, the last; padding pairs join
its first atom with itself through an offset beyond the model's cutoff, so the cutoff
function zeroes their messages and gradients. ``properties.atom_mask`` and
``properties.structure_mask`` mark the real entries (1.0) and the padding (0.0).

Molecules only: no cells or periodic images.
"""

import json
import os

import numpy as np
import torch

from schnetpack import properties
from schnetpack.data.atoms import ASEAtomsData

__all__ = ["convert_to_arrays", "DeviceAtomsData", "ShuffledBatches", "OrderedBatches"]

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


def convert_to_arrays(
    dataset: ASEAtomsData, out_dir: str, num_workers: int = 0, batch_size: int = 1000
) -> dict:
    """
    Read every structure of a dataset into flat arrays (written to out_dir).

    The dataset is read without its transforms, in index order, so array
    position k holds dataset index k. Per-atom properties (first dimension
    n_atoms) are concatenated; per-structure ones (first dimension 1) are
    stacked.

    Args:
        dataset: the dataset
        out_dir: directory for the ``.npy`` files and ``meta.json``
        num_workers: DataLoader workers reading the db
        batch_size: structures per read

    Returns:
        the metadata written to ``meta.json``

    Raises:
        NotImplementedError: for periodic structures
    """
    raw = dataset.subset(list(range(len(dataset))))
    raw.transforms = []
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
    for samples in loader:
        for s in samples:
            if bool(s[properties.pbc].any()):
                raise NotImplementedError("periodic structures are not supported yet")
            n = int(s[properties.n_atoms].reshape(-1)[0])
            for key, value in s.items():
                if key in (properties.idx, properties.cell, properties.pbc):
                    continue
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
    }
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
        self.n_pairs_cpu = self.n_atoms_cpu * (self.n_atoms_cpu - 1)
        self.n_atoms = self.n_atoms_cpu.to(self.device)
        self.n_pairs = self.n_pairs_cpu.to(self.device)
        self.atom_start = torch.cumsum(self.n_atoms, 0) - self.n_atoms
        self.Z = load("Z").to(self.device)
        self.R = load("R").to(self.device)
        self.properties = {
            key: (load(key).to(self.device), kind)
            for key, kind in self.meta["properties"].items()
        }

    def __len__(self) -> int:
        return len(self.n_atoms_cpu)

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
        self, sel: torch.Tensor, n_atoms_pad: int, n_pairs_pad: int, cutoff: float
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
            cutoff: the model's cutoff; padding pairs sit 1 beyond it

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
        # pair slots: every ordered pair i != j of their structure
        k2 = torch.arange(n_pairs_pad, device=dev)
        b2 = torch.searchsorted(cp, k2, right=True)
        real2 = b2 < B
        b2c = b2.clamp(max=B - 1)
        q = k2 - pfirst[b2c]
        nm1 = (n[b2c] - 1).clamp(min=1)
        i = torch.div(q, nm1, rounding_mode="floor")
        r = q - i * nm1
        j = r + (r >= i).long()
        pad_atom = cn[-1:].expand(n_pairs_pad)  # the first padding atom
        offsets = torch.zeros(n_pairs_pad, 3, device=dev, dtype=R_dtype)
        offsets[:, 0] = torch.where(real2, 0.0, cutoff + 1.0)
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
            properties.cell: torch.zeros(B + 1, 3, 3, device=dev, dtype=R_dtype),
            properties.pbc: torch.zeros(3 * (B + 1), dtype=torch.bool, device=dev),
            properties.atom_mask: mask,
            properties.structure_mask: torch.cat(
                [
                    torch.ones(B, device=dev, dtype=R_dtype),
                    torch.zeros(1, device=dev, dtype=R_dtype),
                ]
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

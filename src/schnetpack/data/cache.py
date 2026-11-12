"""
A dataset's structures, computed once and held as flat tensors in host memory.

:meth:`AtomsCache.from_dataset` reads ``dataset[i]`` for every i, with the dataset's
transforms applied, and keeps the results; :class:`~schnetpack.data.AtomsLoader`
with ``cache=`` then slices whole padded batches from it instead of reading the
structures one by one. The transforms that run there must act on one structure and
be deterministic (centering, a neighbor list); noise, offsets and casts go into the
loader's ``batch_transforms``. A cache can be saved to a folder and loaded from it.

One tensor per key, named as in a batch, with the structures back to back, so a
structure is a run of rows of every tensor and any set of structures is gathered
into a batch without reading or decoding anything. For structures of 3, 5 and 2
atoms::

    _n_atoms             = [3, 5, 2]                per structure
    atom_start           = [0, 3, 8]                where each structure's atoms begin
    _atomic_numbers, ... = shape (10, ...)          per-atom keys, all atoms in a row
    _cell, _pbc          = shape (3, 3, 3), (3, 3)  per-structure keys
    _idx_i_local, _idx_j_local, shift               stored pairs: local atoms, cell shifts
    n_pairs, pair_start                             per structure, as for the atoms
    index                                           the db index of each structure
"""

import json
import os
import shutil

import numpy as np
import torch

from schnetpack import properties
from schnetpack.data.stats import fit_atomrefs, structure_stats
from schnetpack.transform.base import Transform

__all__ = ["AtomsCache", "ArrayStats"]

#: version of the folders AtomsCache.save writes (8: the db index of every
#: structure as "index", no preprocess descriptors; 7: torch tensors in one arrays.pt)
ARRAYS_FORMAT = 8


class AtomsCache:
    """
    The structures of a dataset as flat tensors in host memory (module docstring):
    made by :meth:`from_dataset`, or loaded from a folder that :meth:`save` wrote.
    The whole dataset is held: it has to fit into the host's memory. Units and atom
    references go along, so the tensors stand without the dataset. ::

        cache = AtomsCache.from_dataset(dataset, num_workers=8)
        train = AtomsLoader(dataset.subset(train_idx), 100, shuffle=True, cache=cache)
    """

    def __init__(self, arrays: dict[str, torch.Tensor], meta: dict):
        """
        Args:
            arrays: tensors under the batch keys ``properties.n_atoms``, ``Z``,
                ``R``, ``cell``, ``pbc`` (one row of 3 per structure), the stored
                pairs (``properties.lidx_i``, ``lidx_j``: atoms counted within each
                structure; ``shift``: integer cell shifts; ``n_pairs``), ``index``
                (the db index of each structure) and the properties
            meta: units, distance unit, atom references and the kind of each
                property (as :meth:`from_dataset` writes them)
        """
        self.arrays, self.meta = arrays, meta
        self.units = dict(meta["units"])
        self.distance_unit = meta["distance_unit"]
        self._atomrefs = {k: torch.tensor(v) for k, v in meta["atomrefs"].items()}
        #: atoms of each structure, and where they start in the per-atom tensors
        self.n_atoms = arrays[properties.n_atoms]
        self.atom_start = torch.cumsum(self.n_atoms, 0) - self.n_atoms
        self.Z, self.R = arrays[properties.Z], arrays[properties.R]
        #: the db index of the structure at each position
        self.index = arrays["index"]
        #: the other properties: key -> (tensor, "atom" or "structure")
        self.properties = {
            key: (arrays[key], kind) for key, kind in meta["properties"].items()
        }
        self.cell = arrays[properties.cell].to(self.R.dtype)
        self.pbc = arrays[properties.pbc]
        #: the pairs the dataset's neighbor list made (None without): local atoms,
        #: integer cell shifts, the pairs of each structure and where they start
        self.n_pairs = torch.zeros_like(self.n_atoms)
        self.pair_i = self.pair_j = self.shift = self.pair_start = None
        if properties.lidx_i in arrays:
            self.pair_i = arrays[properties.lidx_i]
            self.pair_j = arrays[properties.lidx_j]
            self.shift, self.n_pairs = arrays["shift"], arrays["n_pairs"]
            self.pair_start = torch.cumsum(self.n_pairs, 0) - self.n_pairs
        #: whether any structure has periodic boundary conditions
        self.periodic = bool(self.pbc.any())

    @classmethod
    def load(cls, path: str) -> "AtomsCache":
        """The tensors :meth:`save` wrote to the folder ``path``, as they are: they
        are not checked against a dataset or its transforms."""
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if meta.get("format") != ARRAYS_FORMAT:
            raise ValueError(
                f"the cache in {path} has format {meta.get('format')}, "
                f"this version reads format {ARRAYS_FORMAT}: make it again"
            )
        arrays = torch.load(os.path.join(path, "arrays.pt"), weights_only=True)
        return cls(arrays, meta)

    def save(self, path: str) -> None:
        """
        Write the tensors to the folder ``path``: ``arrays.pt`` (:func:`torch.save`)
        and ``meta.json``. Written as ``<path>.tmp`` and renamed when complete, so a
        killed save leaves no half-written folder. An earlier save there is
        replaced; a folder that holds other files raises.
        """
        path = os.fspath(path).rstrip("/")
        tmp = path + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)  # left by a killed save
        os.makedirs(tmp)
        torch.save(self.arrays, os.path.join(tmp, "arrays.pt"))
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        if os.path.exists(os.path.join(path, "meta.json")):
            shutil.rmtree(path)  # an earlier save
        try:
            os.replace(tmp, path)
        except OSError as e:  # a folder that holds other files
            shutil.rmtree(tmp)
            raise ValueError(f"{path} holds other files: name a new folder") from e

    @classmethod
    def from_dataset(cls, dataset, num_workers: int = 0) -> "AtomsCache":
        """
        ``dataset[i]`` for every i, computed once and held as tensors in memory:
        position k holds ``dataset[k]``. Nothing is written to disk (:meth:`save`
        does that).

        Every structure passes through the dataset's transforms (of an
        :class:`~schnetpack.data.ASEAtomsData`: the list of its split, so make the
        cache before splitting). They must act on one structure and be
        deterministic: a transform that moves the atoms (noise) or needs statistics
        (offsets, scaling) raises, as it belongs into the loader's
        ``batch_transforms``. Of each result the tensors keep the positions, cell
        and pbc, the pairs of a neighbor list (``idx_i``, ``idx_j``, ``offsets``,
        kept as local atoms and integer cell shifts), the db index
        (``properties.idx``) and every other key per atom (first dimension n_atoms)
        or per structure (first dimension 1); any other key raises.

        Args:
            dataset: the structures, e.g. an :class:`~schnetpack.data.ASEAtomsData`
                with ``transforms=[SubtractCenterOfMass(), MatScipyNeighborList(5.0)]``
            num_workers: DataLoader workers reading the dataset
        """
        if hasattr(dataset, "get_split_transforms"):
            transforms = dataset.get_split_transforms()
        else:
            transforms = getattr(dataset, "transforms", None) or []
        for t in transforms:
            for m in t.modules():  # wrappers hold other transforms
                name = type(m).__name__
                if getattr(m, "moves_atoms", False):
                    raise ValueError(
                        f"{name} moves the atoms: put it into the loader's "
                        "batch_transforms"
                    )
                if isinstance(m, Transform) and (
                    type(m).initialize is not Transform.initialize
                ):
                    raise ValueError(
                        f"{name} needs statistics: put it into the loader's "
                        "batch_transforms"
                    )
        if len(dataset) == 0:
            raise ValueError("the dataset holds no structures")
        # workers read the structures with the dataset's transforms; they hand back
        # 1000 structures at a time as one tensor per key (_columns)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1000,
            num_workers=num_workers,
            collate_fn=_columns,
            timeout=600 if num_workers else 0,  # raise rather than hang
        )
        columns: dict[str, list[torch.Tensor]] = {}
        kinds: dict[str, str] = {}  # each property: "atom" or "structure"
        for batch, batch_kinds in loader:
            for key, kind in batch_kinds.items():
                if kind == "atom":  # a structure of one atom looks per structure
                    kinds[key] = kind
                else:
                    kinds.setdefault(key, kind)
            # copied out of shared memory: a tensor from a worker holds a file
            # descriptor while it lives, and the open-file limit (1024 on Hydra)
            # is soon reached
            for key, values in batch.items():
                columns.setdefault(key, []).append(values.clone())
        # one tensor per key: the atoms of all structures back to back, one row per
        # structure for the per-structure keys
        arrays = {key: torch.cat(values) for key, values in columns.items()}
        arrays[properties.n_atoms] = arrays[properties.n_atoms].long()
        arrays[properties.Z] = arrays[properties.Z].long()
        arrays[properties.pbc] = arrays[properties.pbc].bool()
        arrays["index"] = arrays["index"].long()
        n_atoms = arrays[properties.n_atoms]
        if "n_pairs" in arrays:
            if len(arrays["n_pairs"]) != len(n_atoms):
                raise ValueError(
                    "the neighbor list made pairs for some structures only"
                )
            # local atoms fit int16 up to 32767 atoms per structure, int32 above
            small = int(n_atoms.max()) <= torch.iinfo(torch.int16).max
            index = torch.int16 if small else torch.int32
            for key in (properties.lidx_i, properties.lidx_j):
                arrays[key] = arrays[key].to(index)
            arrays["shift"] = arrays["shift"].reshape(-1, 3)
            arrays["n_pairs"] = arrays["n_pairs"].long()
        # what the tensors need besides the numbers to stand alone (saved as
        # meta.json): sizes, the kind and unit of each property
        atomrefs = dataset.atomrefs
        meta = {
            "structures": int(len(n_atoms)),
            "atoms": int(len(arrays[properties.Z])),
            "properties": kinds,
            "units": {key: dataset.units.get(key) for key in kinds},
            "distance_unit": dataset.distance_unit,
            "atomrefs": {
                key: ref.double().tolist()
                for key, ref in atomrefs.items()
                if key in kinds
            },
            "format": ARRAYS_FORMAT,
        }
        return cls(arrays, meta)

    def __len__(self) -> int:
        return len(self.n_atoms)

    @property
    def atomrefs(self) -> dict[str, torch.Tensor]:
        return dict(self._atomrefs)

    def positions(self, dataset) -> np.ndarray:
        """
        Where the structures of a dataset view lie in the cache: position k of the
        result holds ``dataset[k]`` (found by the db index, ``dataset.subset_idx``).

        Raises:
            ValueError: for a structure the cache does not hold
        """
        subset = getattr(dataset, "subset_idx", None)
        wanted = np.asarray(
            range(len(dataset)) if subset is None else subset, dtype=np.int64
        )
        where = np.full(int(self.index.max()) + 1, -1, dtype=np.int64)
        where[self.index.numpy()] = np.arange(len(self))
        held = (wanted >= 0) & (wanted < len(where))
        found = np.full(len(wanted), -1, dtype=np.int64)
        found[held] = where[wanted[held]]
        missing = wanted[found < 0]
        if len(missing):
            raise ValueError(
                f"the cache does not hold the structures {missing[:5].tolist()}"
                f"{' ...' if len(missing) > 5 else ''} (db index): make it from the "
                "dataset the views are taken from"
            )
        return found

    def read_properties(self, keys: list[str], indices=None) -> dict[str, torch.Tensor]:
        """
        Some properties of many structures as flat tensors, with their atom counts,
        the structure (0..) of each atom and the atomic numbers: what
        :func:`~schnetpack.data.structure_stats` and
        :func:`~schnetpack.data.fit_atomrefs` take.

        Args:
            keys: the properties
            indices: positions in the cache (default: all)
        """
        if indices is None:
            indices = range(len(self))
        idx = torch.as_tensor(indices, dtype=torch.long)
        n = self.n_atoms[idx]
        atoms = _rows(self.atom_start[idx], n)
        out = {}
        for key in keys:
            values, kind = self.properties[key]
            out[key] = values[atoms] if kind == "atom" else values[idx]
        out[properties.n_atoms] = n
        out[properties.idx_m] = torch.repeat_interleave(torch.arange(len(idx)), n)
        out[properties.Z] = self.Z[atoms]
        return out

    def gather(
        self,
        sel,
        n_atoms_pad: int | None = None,
        n_pairs_pad: int | None = None,
        n_slots: int | None = None,
        out: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        The batch of some structures, as tensors in host memory. The atoms and
        pairs of a structure lie back to back in the tensors, so each key is
        gathered by its rows straight into the batch's tensor.

        A padded batch has fixed sizes: padding atoms (Z = 0, at the origin) form
        one extra structure, the last, with an identity cell; padding pairs join its
        first atom with itself through an offset of 1000 along x: beyond any
        cutoff, so the cutoff function zeroes their messages, and of nonzero
        length. ``properties.atom_mask`` and ``properties.structure_mask`` mark the
        real entries (1.0) and the padding. ``properties.idx`` holds the db index
        of each real structure and -1 in the free slots.

        The gather runs on one torch thread: its ops are many and small, and
        torch's thread pool only adds waits to them (on a busy CPU, 100 times the
        work).

        Args:
            sel: positions of the structures
            n_atoms_pad: atoms with padding (one padding atom at least). None:
                exact sizes, without padding and masks
            n_pairs_pad: pairs with padding. None: not padded
            n_slots: structure slots of a padded batch; the free ones hold empty
                structures. None: one per structure
            out: tensors of an earlier batch to write into, where they fit

        The pairs are gathered if the tensors have them (the dataset's neighbor
        list made them).
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return self._gather(sel, n_atoms_pad, n_pairs_pad, n_slots, out)
        finally:
            torch.set_num_threads(threads)

    def _gather(self, sel, n_atoms_pad, n_pairs_pad, n_slots, out):
        sel = torch.as_tensor(sel, dtype=torch.long)
        k, n = len(sel), self.n_atoms[sel]
        N, dtype = int(n.sum()), self.R.dtype
        padded = n_atoms_pad is not None
        A = n_atoms_pad if padded else N  # atom rows
        B = k if n_slots is None else n_slots
        S = B + 1 if padded else k  # structure rows; the padding structure is the last
        rows = _rows(self.atom_start[sel], n)  # of the batch's atoms
        batch = {}

        # the tensor of a key: the one in `out` if it fits, else a new one
        def new(key, shape, dtype):
            held = None if out is None else out.get(key)
            if held is None or held.shape != shape or held.dtype != dtype:
                held = torch.empty(shape, dtype=dtype)
            batch[key] = held
            return held

        # a per-atom key: the batch's rows, then 0 for the padding atoms
        def per_atom(key, values):
            x = new(key, (A, *values.shape[1:]), values.dtype)
            torch.index_select(values, 0, rows, out=x[:N])
            x[N:] = 0

        # a per-structure key: the selected rows, then `free` for the free slots and
        # `last` for the padding structure
        def per_structure(key, values, free=0, last=0):
            x = new(key, (S, *values.shape[1:]), values.dtype)
            x[:k] = values
            if padded:
                x[k:B], x[B] = free, last

        # the structures: db index, atoms, the structure of each atom (padding atoms
        # belong to the padding structure B), sizes, cells, pbc, masks
        index = self.index[sel]
        per_structure(properties.idx, index, -1, index[0] if k else -1)
        per_atom(properties.Z, self.Z)
        per_atom(properties.R, self.R)
        idx_m = new(properties.idx_m, (A,), torch.int64)
        idx_m[:N], idx_m[N:] = torch.repeat_interleave(torch.arange(k), n), B
        per_structure(properties.n_atoms, n, 0, A - N)
        # the free slots and the padding structure get a regular cell, so that
        # transforms and models that invert or strain cells work on padded batches
        eye = torch.eye(3, dtype=dtype)
        per_structure(properties.cell, self.cell[sel], eye, eye)
        pbc = new(properties.pbc, (3 * S,), torch.bool)
        pbc[: 3 * k], pbc[3 * k :] = self.pbc[sel].reshape(-1), False
        if padded:
            mask = new(properties.atom_mask, (A,), dtype)
            mask[:N], mask[N:] = 1.0, 0.0
            per_structure(properties.structure_mask, (n > 0).to(dtype), 0.0, 0.0)
        # the properties, by kind
        for key, (values, kind) in self.properties.items():
            if kind == "structure":
                per_structure(key, values[sel])
            else:
                per_atom(key, values)
        if self.pair_i is None:
            return batch

        # the pairs: their local atoms become batch atoms by adding the first atom
        # of their structure, and their offsets are shift @ cell (all zero without
        # periodic structures)
        p = self.n_pairs[sel]
        P = int(p.sum())
        M = P if n_pairs_pad is None else n_pairs_pad
        idx_i = new(properties.idx_i, (M,), torch.int64)
        idx_j = new(properties.idx_j, (M,), torch.int64)
        offsets = new(properties.offsets, (M, 3), dtype)
        offsets[:P] = 0.0
        offsets[P:] = torch.tensor([1000.0, 0.0, 0.0])  # padding: beyond any cutoff
        if P:
            pairs = _rows(self.pair_start[sel], p)  # of the batch's pairs
            first = torch.repeat_interleave(torch.cumsum(n, 0) - n, p)
            torch.add(self.pair_i[pairs], first, out=idx_i[:P])
            torch.add(self.pair_j[pairs], first, out=idx_j[:P])
            if self.periodic:  # periodic images: offset = shift @ cell
                cells = self.cell[sel].repeat_interleave(p, 0)
                images = self.shift[pairs].to(dtype)
                offsets[:P] = torch.einsum("pk,pkd->pd", images, cells)
        idx_i[P:], idx_j[P:] = N, N  # padding pairs: the first padding atom
        return batch


class ArrayStats:
    """
    Statistics and atom references of some structures (the training split) from a
    cache: a stats source (``get_stats``, ``get_atomrefs``) for the transforms and
    the model's postprocessors, as :class:`~schnetpack.data.StatsAtomrefProvider` is
    for the db. No pass over the db (~46 s for QM9); everything in float64. The
    values are those the cache holds, after the dataset's transforms.
    """

    def __init__(
        self,
        cache: AtomsCache,
        indices,
        atomrefs: dict[str, torch.Tensor] | None = None,
    ):
        """
        Args:
            cache: the structures
            indices: positions in the cache, e.g. ``cache.positions(train_view)``
                (of a cache made from the whole dataset: the dataset's indices)
            atomrefs: the atom references (default: the cache's, which are the
                dataset's); properties without one are fitted
        """
        self.cache, self.indices = cache, indices
        self.atomrefs = cache.atomrefs if atomrefs is None else atomrefs

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atomref = self.atomrefs.get(property) if remove_atomref else None
        flat = self.cache.read_properties([property], self.indices)
        return structure_stats(
            flat[property],
            flat[properties.n_atoms],
            flat[properties.idx_m],
            flat[properties.Z],
            divide_by_atoms,
            atomref,
        )

    def get_atomrefs(
        self, property: str, is_extensive: bool, estimate: bool = True
    ) -> dict[str, torch.Tensor]:
        if property in self.atomrefs:
            return {property: self.atomrefs[property]}
        if not estimate:
            raise RuntimeError(
                f"The dataset provides no atomrefs for property '{property}' "
                "and atomref estimation is disabled."
            )
        flat = self.cache.read_properties([property], self.indices)
        fitted = fit_atomrefs(
            flat[property],
            flat[properties.n_atoms],
            flat[properties.idx_m],
            flat[properties.Z],
            is_extensive,
        )
        return {property: fitted}


def _rows(start: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """The rows start[k] .. start[k] + n[k] - 1 of every k, back to back: where the
    atoms (or pairs) of some structures lie in the flat tensors."""
    first = torch.cumsum(n, 0) - n  # of each structure in the result
    return torch.repeat_interleave(start - first, n) + torch.arange(int(n.sum()))


# keys of a structure that are not stored as properties: idx is kept as "index",
# cell, pbc and the pairs are stored apart, and a neighbor list's own local pairs
# (AllToAllNeighborList) are replaced by the stored ones
_NOT_PROPERTIES = {
    properties.idx,
    properties.cell,
    properties.pbc,
    properties.idx_i,
    properties.idx_j,
    properties.offsets,
    properties.lidx_i,
    properties.lidx_j,
}


def _columns(samples):
    """
    In a worker: structures after the dataset's transforms as one tensor per key,
    and the kind of each property ("atom" or "structure"). One tensor per key keeps
    the handoff to the main process to a few tensors, each of which crosses through
    shared memory with a file descriptor.
    """
    rows: dict[str, list[torch.Tensor]] = {}
    kinds: dict[str, str] = {}
    for s in samples:
        cell = s[properties.cell].reshape(3, 3)
        row = {properties.cell: cell[None]}  # one row per structure
        row[properties.pbc] = s[properties.pbc].reshape(1, 3)
        row["index"] = s[properties.idx].reshape(1)
        # pairs of a neighbor list: a single structure's idx_i, idx_j count its own
        # atoms; offsets are stored as integer cell shifts
        if properties.idx_i in s:
            row[properties.lidx_i] = s[properties.idx_i]
            row[properties.lidx_j] = s[properties.idx_j]
            row["shift"] = _cell_shifts(s[properties.offsets], cell)
            row["n_pairs"] = torch.tensor([len(s[properties.idx_i])])
        # every other key is a property, per atom (first dimension n) or per
        # structure (first dimension 1)
        n = int(s[properties.n_atoms].reshape(-1)[0])
        for key, value in s.items():
            if key in _NOT_PROPERTIES:
                continue
            v = value.reshape(-1) if value.ndim == 0 else value
            row[key] = v
            if key in (properties.Z, properties.R, properties.n_atoms):
                continue
            if v.shape[0] == n and n != 1:
                kinds[key] = "atom"
            elif v.shape[0] == 1:
                kinds.setdefault(key, "structure")
            else:
                raise ValueError(
                    f"property {key!r} has shape {tuple(v.shape)}: neither per atom "
                    f"({n} atoms) nor per structure (1)"
                )
        for key, v in row.items():
            rows.setdefault(key, []).append(v)
    return {key: torch.cat(v) for key, v in rows.items()}, kinds


def _cell_shifts(offsets: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """The integer cell shifts S of pair offsets (offset = S @ cell)."""
    if not offsets.any():
        return torch.zeros(offsets.shape, dtype=torch.int8)
    offsets = offsets.to(cell.dtype)
    shifts = torch.linalg.solve(cell.T, offsets.T).T.round()
    if (shifts @ cell - offsets).abs().max() > 1e-6 * max(cell.abs().max(), 1.0):
        raise ValueError("pair offsets are not integer multiples of the cell vectors")
    if shifts.abs().max() > torch.iinfo(torch.int8).max:
        raise ValueError("a pair spans more than 127 cells")
    return shifts.to(torch.int8)

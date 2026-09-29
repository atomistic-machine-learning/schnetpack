"""
HDF5 trajectories of :class:`~schnetpack.dynamics.base.Dynamics` runs.

A driver moves a batch of structures — of any sizes — in lockstep, holding
the atoms of every structure end to end. The file keeps that flat layout and
the structure sizes needed to cut it apart again::

    /                  attrs: n_structures, n_total_atoms, driver, ...,
                              energy_unit, position_unit, schnetpack_version
    /n_atoms           (n_structures,)
    /atomic_numbers    (n_total_atoms,)
    /pbc               (n_structures, 3)                    if the batch has it
    /steps             (n_frames,)                          step per frame
    /positions         (n_frames, n_total_atoms, 3)
    /cell              (n_frames, n_structures, 3, 3)       if the batch has it

plus what the frames of the run carry, e.g.::

    /t                 (n_frames,)                          sampling
    /energy            (n_frames, n_structures)             relaxation
    /converged         (n_frames, n_structures)             relaxation
    /forces            (n_frames, n_total_atoms, 3)         only if store_forces

so that, e.g., ``reader.structure(3)["positions"]`` is the whole path of
structure 3, read off disk without the rest of the file.

This is deliberately not the layout ``schnetpack.md`` writes. That one
carries a replica axis, packs positions, energies, cells and stresses into a
single flat array decoded by hard-coded offsets, and needs the masses and
time step of an MD run — none of which a relaxation or a sampling run has.
"""

from typing import Any

import h5py
import numpy as np
import torch

from schnetpack import properties

__all__ = ["TrajectoryWriter", "TrajectoryReader"]


def _schnetpack_version() -> str:
    """Recorded so a trajectory can be traced back to the code that wrote it."""
    from schnetpack import __version__

    return __version__


def _to_numpy(value: torch.Tensor | np.ndarray, dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class TrajectoryWriter:
    """
    Buffered writer for the layout described in the module docstring.

    Frames are accumulated in memory and written one slab at a time, so a run
    pays for ``n_frames / buffer_size`` HDF5 writes rather than one per
    frame. The frame datasets grow as the run goes on, since a relaxation
    does not know how many steps it will take. Which per-frame datasets
    exist is settled by the first frame written: every quantity it carries
    gets one, and later frames must carry the same.
    """

    #: per-frame quantities with one value per structure (the others are
    #: per atom when their leading axis is n_total_atoms, or scalar)
    _per_structure = ("energy", "converged")

    def __init__(
        self,
        filename: str,
        atomic_numbers: torch.Tensor | np.ndarray,
        n_atoms: torch.Tensor | np.ndarray,
        pbc: torch.Tensor | np.ndarray | None = None,
        store_forces: bool = False,
        buffer_size: int | None = None,
        precision: int = 32,
        attrs: dict | None = None,
        max_buffer_bytes: int = 8 << 20,
    ):
        """
        Args:
            filename: path of the HDF5 file to create
            atomic_numbers: ``(n_total_atoms,)``
            n_atoms: ``(n_structures,)`` atom count of each structure
            pbc: ``(n_structures, 3)`` periodic boundary conditions, if any
            store_forces: also store per-atom forces when frames carry them.
                Off by default — they are as large as the positions.
            buffer_size: frames held in memory before a write, doubling as
                the HDF5 chunk size along the frame axis. ``None`` derives it
                from ``max_buffer_bytes``, which keeps chunks near the size
                HDF5 is happy with whatever the batch size. One frame is the
                floor.
            precision: 32 or 64, the float precision of the stored data
            attrs: extra root attributes, e.g. the driver name
            max_buffer_bytes: byte budget used to derive ``buffer_size``
        """
        if precision not in (32, 64):
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        self.dtype = np.float32 if precision == 32 else np.float64
        self.store_forces = store_forces
        self._buffer_size = buffer_size
        self._max_buffer_bytes = max_buffer_bytes

        n_atoms = _to_numpy(n_atoms, np.int64).reshape(-1)
        atomic_numbers = _to_numpy(atomic_numbers, np.int32).reshape(-1)
        if atomic_numbers.shape[0] != n_atoms.sum():
            raise ValueError(
                f"{atomic_numbers.shape[0]} atomic numbers for structures of "
                f"{n_atoms.sum()} atoms in total"
            )
        self.n_structures = n_atoms.shape[0]
        self.n_total_atoms = int(n_atoms.sum())

        self.file = h5py.File(filename, "w")
        self.file.attrs["n_structures"] = self.n_structures
        self.file.attrs["n_total_atoms"] = self.n_total_atoms
        self.file.attrs["energy_unit"] = "eV"
        self.file.attrs["position_unit"] = "Ang"
        self.file.attrs["schnetpack_version"] = _schnetpack_version()
        for key, value in (attrs or {}).items():
            self.file.attrs[key] = value

        self.file.create_dataset("n_atoms", data=n_atoms)
        self.file.create_dataset("atomic_numbers", data=atomic_numbers)
        if pbc is not None:
            self.file.create_dataset(
                "pbc", data=_to_numpy(pbc, bool).reshape(self.n_structures, 3)
            )

        self.datasets: dict[str, h5py.Dataset] = {}
        self.buffers: dict[str, np.ndarray] = {}
        self.buffer_size = None
        self.n_frames = 0
        self._buffered = 0

    def _frame_shape(self, name: str, value) -> tuple[tuple, Any]:
        """Shape and dtype one frame of ``name`` is stored with."""
        if name == "steps":
            return (), np.int32
        if name == "positions" or name == "forces":
            return (self.n_total_atoms, 3), self.dtype
        if name == "cell":
            return (self.n_structures, 3, 3), self.dtype
        if name == "converged":
            return (self.n_structures,), bool
        if name in self._per_structure:
            return (self.n_structures,), self.dtype
        size = int(np.prod(np.shape(_to_numpy(value, self.dtype))))
        if size == 1:
            return (), self.dtype
        if size == self.n_structures:
            return (self.n_structures,), self.dtype
        raise ValueError(
            f"cannot store {name!r}: one value per frame or per structure is "
            f"expected, got {size} values"
        )

    def _create(self, frame: dict[str, Any]) -> None:
        specs = [(name, *self._frame_shape(name, v)) for name, v in frame.items()]
        buffer_size = self._buffer_size
        if buffer_size is None:
            frame_bytes = sum(
                int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                for _, shape, dtype in specs
            )
            buffer_size = int(
                np.clip(self._max_buffer_bytes // max(frame_bytes, 1), 1, 64)
            )
        self.buffer_size = buffer_size
        for name, shape, dtype in specs:
            self.datasets[name] = self.file.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                chunks=(buffer_size,) + shape,
                dtype=dtype,
            )
            self.buffers[name] = np.zeros((buffer_size,) + shape, dtype=dtype)

    def write(
        self,
        step: int,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
        forces: torch.Tensor | None = None,
        **quantities,
    ) -> None:
        """
        Append one frame. Tensors may live on any device.

        ``positions`` and ``forces`` are taken flat, ``(n_total_atoms, 3)``,
        the way drivers hold them. Further per-frame quantities (``t``,
        ``energy``, ``converged``) come as keywords; ``None`` values are
        skipped. Forces are stored only with ``store_forces``.
        """
        frame = {"steps": step, "positions": positions}
        if cell is not None:
            frame["cell"] = cell
        if forces is not None and self.store_forces:
            frame["forces"] = forces
        frame.update({k: v for k, v in quantities.items() if v is not None})

        if not self.datasets:
            self._create(frame)
        if frame.keys() != self.datasets.keys():
            raise ValueError(
                f"frame carries {sorted(frame)}, but this trajectory stores "
                f"{sorted(self.datasets)}"
            )
        for name, value in frame.items():
            buffer = self.buffers[name]
            buffer[self._buffered] = _to_numpy(value, buffer.dtype).reshape(
                buffer.shape[1:]
            )
        self._buffered += 1
        self.n_frames += 1

        if self._buffered == self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write the buffered frames out and empty the buffer."""
        if self._buffered == 0:
            return
        start = self.n_frames - self._buffered
        for name, dataset in self.datasets.items():
            dataset.resize(self.n_frames, axis=0)
            dataset[start : self.n_frames] = self.buffers[name][: self._buffered]
        self._buffered = 0
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.flush()
            self.file.close()
            self.file = None

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class TrajectoryReader:
    """
    Read-only view on a file written by :class:`TrajectoryWriter`.

    The frame arrays are exposed as h5py datasets rather than numpy arrays,
    so ``reader.positions[-1]`` reads one frame off disk without
    materializing the whole trajectory.
    """

    _static_keys = ("n_atoms", "atomic_numbers", "pbc")

    def __init__(self, filename: str):
        self.file = h5py.File(filename, "r")
        self.n_structures = int(self.file.attrs["n_structures"])
        self.n_atoms = torch.as_tensor(self.file["n_atoms"][:], dtype=torch.long)
        self.offsets = torch.cumsum(self.n_atoms, 0) - self.n_atoms
        self.n_frames = self.file["positions"].shape[0]

    def __getattr__(self, name: str):
        # datasets are reached as attributes: reader.positions, reader.energy, ...
        file = self.__dict__.get("file")
        if file is not None and name in file:
            return file[name]
        raise AttributeError(f"'{name}' was not stored in this trajectory")

    @property
    def has_forces(self) -> bool:
        return "forces" in self.file

    def _frame_names(self) -> list[str]:
        return [n for n in self.file if n not in self._static_keys]

    def frame(self, index: int = -1) -> dict[str, torch.Tensor]:
        """
        A single frame as a schnetpack input batch, defaulting to the last one.

        The result is a complete batch, so it feeds straight into
        :func:`schnetpack.interfaces.ase_interface.batch_to_atoms` or back
        into a driver to continue from that frame. The frame's other
        quantities come along under their usual keys (``energy``, ``forces``,
        ``converged``, ``t``), and its step under ``"step"``.
        """
        batch = {
            properties.n_atoms: self.n_atoms.clone(),
            properties.idx_m: torch.repeat_interleave(
                torch.arange(self.n_structures), self.n_atoms
            ),
            properties.Z: torch.as_tensor(
                self.file["atomic_numbers"][:], dtype=torch.long
            ),
        }
        if "pbc" in self.file:
            batch[properties.pbc] = torch.as_tensor(
                self.file["pbc"][:], dtype=torch.bool
            )
        keys = {
            "positions": properties.R,
            "cell": properties.cell,
            "energy": properties.energy,
            "forces": properties.forces,
            "steps": "step",
        }
        for name in self._frame_names():
            value = self.file[name][index]
            key = keys.get(name, name)
            batch[key] = int(value) if name == "steps" else torch.as_tensor(value)
        return batch

    def structure(self, index: int) -> dict[str, np.ndarray]:
        """
        The whole path of a single structure, as numpy arrays over frames.

        Unlike :meth:`frame` this is not a batch — it spans frames rather than
        structures, and is meant for analysis and plotting.
        """
        start = int(self.offsets[index])
        stop = start + int(self.n_atoms[index])
        structure = {"atomic_numbers": self.file["atomic_numbers"][start:stop]}
        if "pbc" in self.file:
            structure["pbc"] = self.file["pbc"][index]
        for name in self._frame_names():
            dataset = self.file[name]
            if (
                dataset.ndim >= 2
                and dataset.shape[1] == self.n_structures
                and (name not in ("positions", "forces"))
            ):
                structure[name] = dataset[:, index]
            elif name in ("positions", "forces"):
                structure[name] = dataset[:, start:stop]
            else:
                structure[name] = dataset[:]
        return structure

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self) -> "TrajectoryReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

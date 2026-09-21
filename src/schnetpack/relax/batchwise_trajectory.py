"""HDF5 trajectories for batch-wise structure relaxations.

A batch-wise relaxation propagates ``n_structures`` structures of equal size in
lockstep, so every frame is a dense ``(n_structures, n_atoms, ...)`` block. That is
stored here as named, directly sliceable datasets::

    /                  attrs: n_structures, n_atoms, optimizer, fmax,
                              energy_unit, position_unit, schnetpack_version
    /atomic_numbers    (n_structures, n_atoms)
    /pbc               (n_structures, 3)
    /steps             (n_frames,)                          optimizer step per frame
    /positions         (n_frames, n_structures, n_atoms, 3)
    /cell              (n_frames, n_structures, 3, 3)
    /energy            (n_frames, n_structures)
    /converged         (n_frames, n_structures)
    /forces            (n_frames, n_structures, n_atoms, 3) only if store_forces

so that, e.g., ``reader.positions[:, 3]`` is the whole path of structure 3 without
touching the rest of the file.

This is deliberately not the layout ``schnetpack.md`` writes. That one carries a
replica axis, packs positions, energies, cells and stresses into a single flat array
decoded by hard-coded offsets, and needs the masses and time step of an MD run --
none of which a relaxation has.
"""

from typing import Dict, Optional, Union

import h5py
import numpy as np
import torch

from schnetpack import properties

__all__ = ["BatchwiseTrajectoryWriter", "BatchwiseTrajectoryReader"]


def _schnetpack_version() -> str:
    """Recorded so a trajectory can be traced back to the code that wrote it."""
    from schnetpack import __version__

    return __version__


def _to_numpy(value: Union[torch.Tensor, np.ndarray], dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class BatchwiseTrajectoryWriter:
    """Buffered writer for the layout described in the module docstring.

    Frames are accumulated in memory and written one slab at a time, so a relaxation
    pays for ``n_frames / buffer_size`` HDF5 writes rather than one per frame. The
    frame datasets grow as the run goes on, since a relaxation does not know how many
    steps it will take.

    Args:
        filename: path of the HDF5 file to create.
        atomic_numbers: (n_structures, n_atoms) or (n_structures * n_atoms,).
        pbc: (n_structures, 3) periodic boundary conditions.
        store_forces: also store the forces of every frame. Off by default -- they are
            as large as the positions, and the forces of the final frame come back from
            the optimizer anyway.
        buffer_size: frames held in memory before a write, doubling as the HDF5 chunk
            size along the frame axis. ``None`` derives it from ``max_buffer_bytes``,
            which keeps chunks near the size HDF5 is happy with whatever the batch
            size -- 64 frames of a 256 x 1000 atom batch would be a 200 MB chunk, two
            orders of magnitude above the recommended maximum and a 200 MB read for
            any caller that wanted a single frame. One frame is the floor, so a batch
            whose single frame already exceeds the budget still gets a chunk that big.
        max_buffer_bytes: byte budget used to derive ``buffer_size``.
        precision: 32 or 64, the float precision of the stored data.
        attrs: extra root attributes, e.g. the optimizer name and the force criterion.
    """

    def __init__(
        self,
        filename: str,
        atomic_numbers: Union[torch.Tensor, np.ndarray],
        pbc: Union[torch.Tensor, np.ndarray],
        store_forces: bool = False,
        buffer_size: Optional[int] = None,
        precision: int = 32,
        attrs: Optional[Dict] = None,
        max_buffer_bytes: int = 8 << 20,
    ):
        if precision not in (32, 64):
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        self.dtype = np.float32 if precision == 32 else np.float64
        self.store_forces = store_forces

        pbc = _to_numpy(pbc, bool).reshape(-1, 3)
        self.n_structures = pbc.shape[0]
        atomic_numbers = _to_numpy(atomic_numbers, np.int32).reshape(
            self.n_structures, -1
        )
        self.n_atoms = atomic_numbers.shape[1]

        self.file = h5py.File(filename, "w")
        self.file.attrs["n_structures"] = self.n_structures
        self.file.attrs["n_atoms"] = self.n_atoms
        self.file.attrs["energy_unit"] = "eV"
        self.file.attrs["position_unit"] = "Ang"
        self.file.attrs["schnetpack_version"] = _schnetpack_version()
        for key, value in (attrs or {}).items():
            self.file.attrs[key] = value

        self.file.create_dataset("atomic_numbers", data=atomic_numbers)
        self.file.create_dataset("pbc", data=pbc)

        # (name, per-frame shape, dtype)
        specs = [
            ("steps", (), np.int32),
            ("positions", (self.n_structures, self.n_atoms, 3), self.dtype),
            ("cell", (self.n_structures, 3, 3), self.dtype),
            ("energy", (self.n_structures,), self.dtype),
            ("converged", (self.n_structures,), bool),
        ]
        if store_forces:
            specs.append(("forces", (self.n_structures, self.n_atoms, 3), self.dtype))

        if buffer_size is None:
            frame_bytes = sum(
                int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                for _, shape, dtype in specs
            )
            buffer_size = int(np.clip(max_buffer_bytes // max(frame_bytes, 1), 1, 64))
        self.buffer_size = buffer_size

        self.datasets = {}
        self.buffers = {}
        for name, shape, dtype in specs:
            self.datasets[name] = self.file.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                chunks=(buffer_size,) + shape,
                dtype=dtype,
            )
            self.buffers[name] = np.zeros((buffer_size,) + shape, dtype=dtype)

        self.n_frames = 0
        self._buffered = 0

    def write(
        self,
        step: int,
        positions: torch.Tensor,
        cell: torch.Tensor,
        energy: Optional[torch.Tensor] = None,
        forces: Optional[torch.Tensor] = None,
        converged: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one frame. Tensors may live on any device.

        ``positions`` and ``forces`` are taken flat, ``(n_structures * n_atoms, 3)``,
        the way the optimizer holds them.
        """
        shape = (self.n_structures, self.n_atoms, 3)
        frame = {
            "steps": step,
            "positions": _to_numpy(positions, self.dtype).reshape(shape),
            "cell": _to_numpy(cell, self.dtype).reshape(self.n_structures, 3, 3),
            "energy": (
                np.zeros(self.n_structures, self.dtype)
                if energy is None
                else _to_numpy(energy, self.dtype).reshape(self.n_structures)
            ),
            "converged": (
                np.zeros(self.n_structures, bool)
                if converged is None
                else _to_numpy(converged, bool).reshape(self.n_structures)
            ),
        }
        if self.store_forces:
            frame["forces"] = (
                np.zeros(shape, self.dtype)
                if forces is None
                else _to_numpy(forces, self.dtype).reshape(shape)
            )

        for name, value in frame.items():
            self.buffers[name][self._buffered] = value
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

    def __enter__(self) -> "BatchwiseTrajectoryWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class BatchwiseTrajectoryReader:
    """Read-only view on a file written by ``BatchwiseTrajectoryWriter``.

    The frame arrays are exposed as h5py datasets rather than numpy arrays, so
    ``reader.positions[:, 3]`` reads one structure's path off disk without
    materializing the whole trajectory.
    """

    _frame_keys = ("steps", "positions", "cell", "energy", "converged", "forces")

    def __init__(self, filename: str):
        self.file = h5py.File(filename, "r")
        self.n_structures = int(self.file.attrs["n_structures"])
        self.n_atoms = int(self.file.attrs["n_atoms"])
        self.n_frames = self.file["positions"].shape[0]

    def __getattr__(self, name: str):
        # datasets are reached as attributes: reader.positions, reader.energy, ...
        if name in self._frame_keys or name in ("atomic_numbers", "pbc"):
            try:
                return self.__dict__["file"][name]
            except KeyError:
                raise AttributeError(
                    f"'{name}' was not stored in this trajectory"
                ) from None
        raise AttributeError(name)

    @property
    def has_forces(self) -> bool:
        return "forces" in self.file

    def frame(self, index: int = -1) -> Dict[str, torch.Tensor]:
        """A single frame as a schnetpack input batch, defaulting to the last one.

        The result is a complete batch, so it feeds straight into
        :func:`schnetpack.interfaces.ase_interface.batch_to_atoms` or back into an
        optimizer to continue from that frame. Energies, forces and the convergence
        flags of the frame come along under their usual property keys.
        """
        n_structures, n_atoms = self.n_structures, self.n_atoms
        batch = {
            properties.n_atoms: torch.full((n_structures,), n_atoms, dtype=torch.long),
            properties.idx_m: torch.repeat_interleave(
                torch.arange(n_structures), n_atoms
            ),
            properties.Z: torch.as_tensor(
                self.file["atomic_numbers"][:], dtype=torch.long
            ).view(-1),
            properties.pbc: torch.as_tensor(self.file["pbc"][:], dtype=torch.bool),
            properties.R: torch.as_tensor(self.file["positions"][index]).view(-1, 3),
            properties.cell: torch.as_tensor(self.file["cell"][index]),
        }
        batch[properties.energy] = torch.as_tensor(self.file["energy"][index])
        batch["converged"] = torch.as_tensor(self.file["converged"][index])
        batch["step"] = int(self.file["steps"][index])
        if self.has_forces:
            batch[properties.forces] = torch.as_tensor(self.file["forces"][index]).view(
                -1, 3
            )
        return batch

    def structure(self, index: int) -> Dict[str, np.ndarray]:
        """The whole path of a single structure, as numpy arrays over frames.

        Unlike :meth:`frame` this is not a batch -- it spans frames rather than
        structures, and is meant for analysis and plotting.
        """
        structure = {
            "atomic_numbers": self.file["atomic_numbers"][index],
            "pbc": self.file["pbc"][index],
            "steps": self.file["steps"][:],
        }
        for key in ("positions", "cell", "energy", "converged", "forces"):
            if key in self.file:
                structure[key] = self.file[key][:, index]
        return structure

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self) -> "BatchwiseTrajectoryReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

"""Round trip through ``BatchwiseTrajectoryWriter`` / ``BatchwiseTrajectoryReader``.

The optimizer's own trajectory tests cover which frames get written. These cover the
storage itself: that the values survive, that buffering is invisible to the reader, and
that the optional datasets really are optional.
"""

import numpy as np
import pytest
import torch

from schnetpack import properties
from schnetpack.interfaces.ase_interface import batch_to_atoms
from schnetpack.relax.batchwise_trajectory import (
    BatchwiseTrajectoryReader,
    BatchwiseTrajectoryWriter,
)

N_STRUCTURES = 3
N_ATOMS = 4


def make_frame(step: int, rng: np.random.Generator) -> dict:
    return {
        "step": step,
        "positions": torch.tensor(
            rng.normal(size=(N_STRUCTURES * N_ATOMS, 3)), dtype=torch.float64
        ),
        "cell": torch.tensor(
            rng.normal(size=(N_STRUCTURES, 3, 3)), dtype=torch.float64
        ),
        "energy": torch.tensor(rng.normal(size=N_STRUCTURES), dtype=torch.float64),
        "forces": torch.tensor(
            rng.normal(size=(N_STRUCTURES * N_ATOMS, 3)), dtype=torch.float64
        ),
        "converged": torch.tensor([step % 2 == 0] * N_STRUCTURES),
    }


def write_frames(path, n_frames, buffer_size=64, store_forces=True, precision=32):
    rng = np.random.default_rng(0)
    frames = [make_frame(step, rng) for step in range(n_frames)]
    with BatchwiseTrajectoryWriter(
        path,
        atomic_numbers=torch.full((N_STRUCTURES, N_ATOMS), 6),
        pbc=torch.zeros(N_STRUCTURES, 3, dtype=torch.bool),
        store_forces=store_forces,
        buffer_size=buffer_size,
        precision=precision,
        attrs={"optimizer": "BatchwiseLBFGS", "fmax": 0.05},
    ) as writer:
        for frame in frames:
            writer.write(**frame)
    return frames


def test_values_survive_the_round_trip(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=5)

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.n_frames == 5
        assert traj.n_structures == N_STRUCTURES
        assert traj.n_atoms == N_ATOMS
        assert list(traj.steps) == [0, 1, 2, 3, 4]

        for index, frame in enumerate(frames):
            expected = frame["positions"].numpy().reshape(N_STRUCTURES, N_ATOMS, 3)
            np.testing.assert_allclose(traj.positions[index], expected, rtol=1e-6)
            np.testing.assert_allclose(
                traj.cell[index], frame["cell"].numpy(), rtol=1e-6
            )
            np.testing.assert_allclose(
                traj.energy[index], frame["energy"].numpy(), rtol=1e-6
            )
            assert (traj.converged[index] == frame["converged"].numpy()).all()


@pytest.mark.parametrize("n_frames", [1, 3, 4, 5, 9])
def test_buffering_is_invisible(tmp_path, n_frames):
    """Frame counts either side of a buffer boundary must read back identically."""
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=n_frames, buffer_size=4)

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.n_frames == n_frames
        assert list(traj.steps) == list(range(n_frames))
        np.testing.assert_allclose(
            traj.positions[-1],
            frames[-1]["positions"].numpy().reshape(N_STRUCTURES, N_ATOMS, 3),
            rtol=1e-6,
        )


def test_forces_are_optional(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    write_frames(path, n_frames=3, store_forces=False)

    with BatchwiseTrajectoryReader(path) as traj:
        assert not traj.has_forces
        with pytest.raises(AttributeError, match="not stored"):
            traj.forces


def test_metadata_is_stored(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    write_frames(path, n_frames=2)

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.file.attrs["optimizer"] == "BatchwiseLBFGS"
        assert traj.file.attrs["fmax"] == pytest.approx(0.05)
        assert traj.file.attrs["energy_unit"] == "eV"
        assert traj.file.attrs["schnetpack_version"]
        assert traj.atomic_numbers.shape == (N_STRUCTURES, N_ATOMS)
        assert traj.pbc.shape == (N_STRUCTURES, 3)


def test_a_frame_reads_back_as_an_input_batch(tmp_path):
    """``frame`` must produce a batch that batch_to_atoms and an optimizer accept."""
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=6)

    with BatchwiseTrajectoryReader(path) as traj:
        last = traj.frame()

    assert last[properties.n_atoms].tolist() == [N_ATOMS] * N_STRUCTURES
    assert last[properties.idx_m].tolist() == sum(
        ([idx] * N_ATOMS for idx in range(N_STRUCTURES)), []
    )
    assert last[properties.Z].shape == (N_STRUCTURES * N_ATOMS,)
    assert last[properties.cell].shape == (N_STRUCTURES, 3, 3)
    np.testing.assert_allclose(
        last[properties.R].numpy(), frames[-1]["positions"].numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        last[properties.forces].numpy(), frames[-1]["forces"].numpy(), rtol=1e-6
    )
    assert last["step"] == 5

    # and it really is convertible, off-diagonal cells included
    structures = batch_to_atoms(last)
    assert [len(s) for s in structures] == [N_ATOMS] * N_STRUCTURES
    np.testing.assert_allclose(
        structures[1].cell[:], frames[-1]["cell"].numpy()[1], rtol=1e-6
    )


def test_structure_gives_one_path_across_frames(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    write_frames(path, n_frames=6)

    with BatchwiseTrajectoryReader(path) as traj:
        path_of_one = traj.structure(1)

    assert path_of_one["positions"].shape == (6, N_ATOMS, 3)
    assert path_of_one["energy"].shape == (6,)
    assert path_of_one["atomic_numbers"].shape == (N_ATOMS,)
    assert list(path_of_one["steps"]) == list(range(6))


def test_double_precision_is_kept(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=2, precision=64)

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.positions.dtype == np.float64
        np.testing.assert_allclose(
            traj.positions[0],
            frames[0]["positions"].numpy().reshape(N_STRUCTURES, N_ATOMS, 3),
            rtol=1e-12,
        )


def test_unknown_precision_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        BatchwiseTrajectoryWriter(
            str(tmp_path / "traj.hdf5"),
            atomic_numbers=torch.full((N_STRUCTURES, N_ATOMS), 6),
            pbc=torch.zeros(N_STRUCTURES, 3, dtype=torch.bool),
            precision=16,
        )


@pytest.mark.parametrize(
    "n_structures, n_atoms, expected",
    [(3, 4, 64), (256, 1000, 2), (2000, 5000, 1)],
)
def test_buffer_size_follows_the_batch_size(tmp_path, n_structures, n_atoms, expected):
    """A fixed 64-frame buffer would be a 200 MB chunk for a large batch.

    HDF5 wants chunks well under a megabyte, and a chunk that large also means any
    caller reading a single frame pays for all 64.
    """
    writer = BatchwiseTrajectoryWriter(
        str(tmp_path / "traj.hdf5"),
        atomic_numbers=torch.full((n_structures, n_atoms), 6),
        pbc=torch.zeros(n_structures, 3, dtype=torch.bool),
    )
    assert writer.buffer_size == expected
    assert writer.datasets["positions"].chunks[0] == expected
    writer.close()


def test_explicit_buffer_size_wins(tmp_path):
    writer = BatchwiseTrajectoryWriter(
        str(tmp_path / "traj.hdf5"),
        atomic_numbers=torch.full((N_STRUCTURES, N_ATOMS), 6),
        pbc=torch.zeros(N_STRUCTURES, 3, dtype=torch.bool),
        buffer_size=7,
    )
    assert writer.buffer_size == 7
    writer.close()

"""Round trip through ``TrajectoryWriter`` / ``TrajectoryReader``.

The drivers' own trajectory tests cover which frames get written. These cover the
storage itself: that the values survive for batches of differently sized structures,
that buffering is invisible to the reader, and that the optional datasets really are
optional.
"""

import numpy as np
import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    Calculator,
    FrameCollector,
    Heun,
    Sampler,
    TrajectoryReader,
    TrajectoryRecorder,
    TrajectoryWriter,
)
from schnetpack.generative import VP, VelocityParametrization
from schnetpack.interfaces.ase_interface import batch_to_atoms

N_ATOMS = [2, 4, 3]
N_STRUCTURES = len(N_ATOMS)
N_TOTAL = sum(N_ATOMS)
OFFSETS = np.cumsum([0] + N_ATOMS)


def make_frame(step: int, rng: np.random.Generator) -> dict:
    return {
        "step": step,
        "positions": torch.tensor(rng.normal(size=(N_TOTAL, 3)), dtype=torch.float64),
        "cell": torch.tensor(
            rng.normal(size=(N_STRUCTURES, 3, 3)), dtype=torch.float64
        ),
        "energy": torch.tensor(rng.normal(size=N_STRUCTURES), dtype=torch.float64),
        "forces": torch.tensor(rng.normal(size=(N_TOTAL, 3)), dtype=torch.float64),
        "converged": torch.tensor([step % 2 == 0] * N_STRUCTURES),
    }


def make_writer(path, **kwargs) -> TrajectoryWriter:
    return TrajectoryWriter(
        path,
        atomic_numbers=torch.full((N_TOTAL,), 6),
        n_atoms=torch.tensor(N_ATOMS),
        pbc=torch.zeros(N_STRUCTURES, 3, dtype=torch.bool),
        **kwargs,
    )


def write_frames(path, n_frames, buffer_size=64, store_forces=True, precision=32):
    rng = np.random.default_rng(0)
    frames = [make_frame(step, rng) for step in range(n_frames)]
    with make_writer(
        path,
        store_forces=store_forces,
        buffer_size=buffer_size,
        precision=precision,
        attrs={"driver": "Relaxer", "fmax": 0.05},
    ) as writer:
        for frame in frames:
            writer.write(**frame)
    return frames


def test_values_survive_the_round_trip(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=5)

    with TrajectoryReader(path) as traj:
        assert traj.n_frames == 5
        assert traj.n_structures == N_STRUCTURES
        assert traj.n_atoms.tolist() == N_ATOMS
        assert list(traj.steps) == [0, 1, 2, 3, 4]

        for index, frame in enumerate(frames):
            np.testing.assert_allclose(
                traj.positions[index], frame["positions"].numpy(), rtol=1e-6
            )
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

    with TrajectoryReader(path) as traj:
        assert traj.n_frames == n_frames
        assert list(traj.steps) == list(range(n_frames))
        np.testing.assert_allclose(
            traj.positions[-1], frames[-1]["positions"].numpy(), rtol=1e-6
        )


def test_forces_are_optional(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    write_frames(path, n_frames=3, store_forces=False)

    with TrajectoryReader(path) as traj:
        assert not traj.has_forces
        with pytest.raises(AttributeError, match="not stored"):
            _ = traj.forces


def test_only_what_the_frames_carry_is_stored(tmp_path):
    """A sampling run has no energies or convergence flags, but a time."""
    path = str(tmp_path / "traj.hdf5")
    with make_writer(path) as writer:
        for step in range(3):
            writer.write(
                step=step, positions=torch.zeros(N_TOTAL, 3), t=torch.tensor(1.0 - step)
            )

    with TrajectoryReader(path) as traj:
        assert "energy" not in traj.file and "converged" not in traj.file
        assert "cell" not in traj.file
        assert list(traj.t) == [1.0, 0.0, -1.0]


def test_frames_must_carry_the_same_quantities(tmp_path):
    with make_writer(str(tmp_path / "traj.hdf5")) as writer:
        writer.write(step=0, positions=torch.zeros(N_TOTAL, 3), t=torch.tensor(1.0))
        with pytest.raises(ValueError, match="carries"):
            writer.write(step=1, positions=torch.zeros(N_TOTAL, 3))


def test_atom_counts_must_match(tmp_path):
    with pytest.raises(ValueError, match="atomic numbers"):
        TrajectoryWriter(
            str(tmp_path / "traj.hdf5"),
            atomic_numbers=torch.full((N_TOTAL + 1,), 6),
            n_atoms=torch.tensor(N_ATOMS),
        )


def test_metadata_is_stored(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    write_frames(path, n_frames=2)

    with TrajectoryReader(path) as traj:
        assert traj.file.attrs["driver"] == "Relaxer"
        assert traj.file.attrs["fmax"] == pytest.approx(0.05)
        assert traj.file.attrs["energy_unit"] == "eV"
        assert traj.file.attrs["schnetpack_version"]
        assert traj.atomic_numbers.shape == (N_TOTAL,)
        assert traj.pbc.shape == (N_STRUCTURES, 3)


def test_a_frame_reads_back_as_an_input_batch(tmp_path):
    """``frame`` must produce a batch that batch_to_atoms and a driver accept."""
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=6)

    with TrajectoryReader(path) as traj:
        last = traj.frame()

    assert last[properties.n_atoms].tolist() == N_ATOMS
    assert last[properties.idx_m].tolist() == sum(
        ([idx] * n for idx, n in enumerate(N_ATOMS)), []
    )
    assert last[properties.Z].shape == (N_TOTAL,)
    assert last[properties.cell].shape == (N_STRUCTURES, 3, 3)
    np.testing.assert_allclose(
        last[properties.R].numpy(), frames[-1]["positions"].numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        last[properties.forces].numpy(), frames[-1]["forces"].numpy(), rtol=1e-6
    )
    assert last["step"] == 5

    structures = batch_to_atoms(last)
    assert [len(s) for s in structures] == N_ATOMS
    np.testing.assert_allclose(
        structures[1].cell[:], frames[-1]["cell"].numpy()[1], rtol=1e-6
    )


def test_structure_gives_one_path_across_frames(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=6)

    with TrajectoryReader(path) as traj:
        path_of_one = traj.structure(1)

    assert path_of_one["positions"].shape == (6, N_ATOMS[1], 3)
    np.testing.assert_allclose(
        path_of_one["positions"][-1],
        frames[-1]["positions"].numpy()[OFFSETS[1] : OFFSETS[2]],
        rtol=1e-6,
    )
    assert path_of_one["forces"].shape == (6, N_ATOMS[1], 3)
    assert path_of_one["energy"].shape == (6,)
    assert path_of_one["cell"].shape == (6, 3, 3)
    assert path_of_one["atomic_numbers"].shape == (N_ATOMS[1],)
    assert list(path_of_one["steps"]) == list(range(6))


def test_double_precision_is_kept(tmp_path):
    path = str(tmp_path / "traj.hdf5")
    frames = write_frames(path, n_frames=2, precision=64)

    with TrajectoryReader(path) as traj:
        assert traj.positions.dtype == np.float64
        np.testing.assert_allclose(
            traj.positions[0], frames[0]["positions"].numpy(), rtol=1e-12
        )


def test_unknown_precision_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        make_writer(str(tmp_path / "traj.hdf5"), precision=16)


@pytest.mark.parametrize(
    "n_structures, n_atoms, expected",
    [(3, 4, 64), (256, 1000, 2), (2000, 5000, 1)],
)
def test_buffer_size_follows_the_batch_size(tmp_path, n_structures, n_atoms, expected):
    """A fixed 64-frame buffer would be a 200 MB chunk for a large batch.

    HDF5 wants chunks well under a megabyte, and a chunk that large also means any
    caller reading a single frame pays for all 64.
    """
    writer = TrajectoryWriter(
        str(tmp_path / "traj.hdf5"),
        atomic_numbers=torch.full((n_structures * n_atoms,), 6),
        n_atoms=torch.full((n_structures,), n_atoms),
    )
    writer.write(step=0, positions=torch.zeros(n_structures * n_atoms, 3))
    assert writer.buffer_size == expected
    assert writer.datasets["positions"].chunks[0] == expected
    writer.close()


def test_explicit_buffer_size_wins(tmp_path):
    writer = make_writer(str(tmp_path / "traj.hdf5"), buffer_size=7)
    writer.write(step=0, positions=torch.zeros(N_TOTAL, 3))
    assert writer.buffer_size == 7
    writer.close()


# ----------------------------------------------------- the reverse-diffusion trajectory


def sampling_batch():
    return {
        properties.R: torch.randn(N_TOTAL, 3),
        properties.Z: torch.full((N_TOTAL,), 6),
        properties.n_atoms: torch.tensor(N_ATOMS),
        properties.idx_m: torch.repeat_interleave(
            torch.arange(N_STRUCTURES), torch.tensor(N_ATOMS)
        ),
    }


def zero_velocity(batch):
    return {"prediction": torch.zeros_like(batch[properties.R])}


def test_the_sampler_records_its_reverse_diffusion_path(tmp_path):
    """A ragged generative batch, one frame per step, with the path time."""
    path = str(tmp_path / "sample.hdf5")
    collector = FrameCollector()
    sampler = Sampler(
        Calculator(zero_velocity),
        VP(),
        VelocityParametrization(),
        Heun(),
        churn=0.0,
        observers=[TrajectoryRecorder(path, interval=1), collector],
    )
    out = sampler.denoise(sampling_batch(), 4)

    assert collector.steps == [0, 1, 2, 3, 4]
    assert collector.frames[-1].final and not collector.frames[0].final
    with TrajectoryReader(path) as traj:
        assert traj.n_frames == 5
        assert traj.file.attrs["driver"] == "Sampler"
        assert traj.file.attrs["integrator"] == "Heun"
        # time runs from t_max down to t_min
        assert np.all(np.diff(traj.t[:]) < 0)
        np.testing.assert_allclose(
            traj.positions[-1], out[properties.R].numpy(), rtol=1e-6
        )
        assert traj.structure(1)["positions"].shape == (5, N_ATOMS[1], 3)


def test_an_unobserved_sampler_builds_no_frames(monkeypatch):
    import schnetpack.dynamics.sampling.sampler as sampler_module

    built = []
    monkeypatch.setattr(sampler_module, "SamplingFrame", lambda **kw: built.append(kw))
    sampler = Sampler(
        Calculator(zero_velocity), VP(), VelocityParametrization(), Heun(), churn=0.0
    )
    sampler.denoise(sampling_batch(), 3)
    assert built == []

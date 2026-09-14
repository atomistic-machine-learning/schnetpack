"""Behaviour of ``BatchwiseLBFGS`` that does not need a trained model.

``test_bw_optimizer.py`` checks that a relaxation lands where ase's ``LBFGS`` lands.
The mechanics around it -- what comes back out of the batch, which atoms are allowed
to move, what an invalid batch does -- are cheaper and clearer to pin down against an
analytic potential, which is what this module does.
"""

from typing import Dict, List, Optional

import numpy as np
import pytest
import torch
from ase import Atoms

from schnetpack import properties
from schnetpack.interfaces.ase_interface import atoms_to_batch, batch_to_atoms
from schnetpack.relax.batchwise_optimization import BatchwiseLBFGS
from schnetpack.relax.batchwise_trajectory import BatchwiseTrajectoryReader
from schnetpack.relax.observers import FrameCollector, Interval


class HarmonicCalculator:
    """Every atom is pulled towards the origin by a spring.

    Stands in for a ``BatchwiseCalculator``: the optimizer only ever asks it for
    forces, and the minimum is known exactly.
    """

    def __init__(self, spring_constant: float = 1.0):
        self.spring_constant = spring_constant
        self.device = torch.device("cpu")
        self.results = {}

    def get_forces(
        self,
        inputs: Dict[str, torch.Tensor],
        fixed_atoms_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        forces = -self.spring_constant * inputs[properties.R]
        self.results = {
            "energy": self.get_potential_energy(inputs),
            "forces": forces,
        }
        if fixed_atoms_mask is not None:
            forces = forces[fixed_atoms_mask]
        return forces

    def get_potential_energy(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # one energy per structure, the way a BatchwiseCalculator reports them
        per_atom = 0.5 * self.spring_constant * inputs[properties.R].pow(2).sum(-1)
        counts = inputs[properties.n_atoms].tolist()
        return torch.stack([chunk.sum() for chunk in torch.split(per_atom, counts)])


def make_inputs(
    n_atoms_per_config: List[int],
    cell: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """A schnetpack input batch, built without a converter or a neighbor list."""
    n_configs = len(n_atoms_per_config)
    n_total = sum(n_atoms_per_config)
    rng = np.random.default_rng(seed)

    cells = torch.zeros(n_configs, 3, 3, dtype=torch.float32)
    if cell is not None:
        cells[:] = torch.as_tensor(cell, dtype=torch.float32)

    return {
        properties.n_atoms: torch.tensor(n_atoms_per_config),
        properties.R: torch.tensor(
            rng.normal(scale=1.0, size=(n_total, 3)), dtype=torch.float32
        ),
        properties.Z: torch.full((n_total,), 6, dtype=torch.long),
        properties.cell: cells,
        properties.pbc: torch.full((n_configs, 3), cell is not None),
        properties.idx_m: torch.repeat_interleave(
            torch.arange(n_configs), torch.tensor(n_atoms_per_config)
        ),
    }


def make_optimizer(inputs, **kwargs) -> BatchwiseLBFGS:
    kwargs.setdefault("logfile", None)
    return BatchwiseLBFGS(calculator=HarmonicCalculator(), inputs=inputs, **kwargs)


def test_full_cell_survives_the_round_trip():
    """A triclinic cell must come back as it went in, off-diagonal entries included."""
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.5, 1.0, 5.0]])
    optimizer = make_optimizer(make_inputs([4, 4], cell=cell))

    relaxed, _ = optimizer.get_relaxation_results()

    for structure in batch_to_atoms(relaxed):
        np.testing.assert_allclose(structure.cell[:], cell, atol=1e-5)
        assert structure.pbc.all()


def test_no_mask_matches_an_all_free_mask():
    """``fixed_atoms_mask=None`` must mean the same as "no atom is fixed"."""
    without = make_optimizer(make_inputs([5, 5]))
    without.run(fmax=1e-3, steps=20)

    explicit = make_optimizer(make_inputs([5, 5]), fixed_atoms_mask=[False] * 10)
    explicit.run(fmax=1e-3, steps=20)

    np.testing.assert_allclose(
        without.inputs[properties.R].numpy(),
        explicit.inputs[properties.R].numpy(),
        atol=1e-6,
    )


def test_fixed_atoms_do_not_move():
    inputs = make_inputs([5, 5])
    initial = inputs[properties.R].clone()
    fixed = [True, False, False, False, False] * 2

    optimizer = make_optimizer(inputs, fixed_atoms_mask=fixed)
    optimizer.run(fmax=1e-3, steps=20)

    moved = (inputs[properties.R] - initial).abs().max(dim=1).values
    assert torch.allclose(moved[torch.tensor(fixed)], torch.zeros(2))
    assert (moved[~torch.tensor(fixed)] > 1e-3).all()


def test_fixed_atoms_are_left_out_of_the_convergence_check():
    """A fixed atom far from the minimum must not keep the relaxation running."""
    inputs = make_inputs([2])
    # the fixed atom carries a large force that can never be relaxed away
    inputs[properties.R] = torch.tensor([[50.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    optimizer = make_optimizer(inputs, fixed_atoms_mask=[True, False])
    assert optimizer.run(fmax=0.05, steps=50)


def test_ragged_batches_are_rejected():
    with pytest.raises(ValueError, match="same number of atoms"):
        make_optimizer(make_inputs([3, 4]))


def test_mask_length_is_checked():
    with pytest.raises(ValueError, match="one per atom"):
        make_optimizer(make_inputs([3, 3]), fixed_atoms_mask=[False] * 5)


def test_run_without_a_step_limit_still_takes_steps():
    """``max_steps`` used to default to 0, so ``run(fmax=...)`` did nothing."""
    optimizer = make_optimizer(make_inputs([4]))

    assert optimizer.run(fmax=1e-3)
    assert optimizer.nsteps > 0


def test_run_reports_failure_when_the_step_limit_is_hit():
    optimizer = make_optimizer(make_inputs([4]))

    assert not optimizer.run(fmax=1e-12, steps=2)
    assert optimizer.nsteps == 2


# ---------------------------------------------------------------- the observer seam


def test_interval_zero_is_endpoints_only():
    interval = Interval(0)
    assert interval.due(0, final=False)
    assert not any(interval.due(step, final=False) for step in range(1, 10))
    assert interval.due(7, final=True)


def test_interval_one_is_every_step():
    interval = Interval(1)
    assert all(interval.due(step, final=False) for step in range(10))


def test_interval_n_takes_every_nth_plus_the_endpoints():
    interval = Interval(3)
    due = [step for step in range(10) if interval.due(step, final=False)]
    assert due == [0, 3, 6, 9]
    assert interval.due(
        7, final=True
    ), "the last frame is recorded whatever the interval"


def test_observers_only_see_the_steps_they_asked_for():
    every = FrameCollector(interval=1)
    endpoints = FrameCollector(interval=0)
    optimizer = make_optimizer(make_inputs([2]), observers=[every, endpoints])
    optimizer.run(fmax=1e-6, steps=6)

    assert every.steps == list(range(optimizer.nsteps + 1))
    assert endpoints.steps == [0, optimizer.nsteps]


def test_a_frame_describes_the_state_it_was_taken_from():
    collector = FrameCollector(interval=1)
    inputs = make_inputs([3, 3])
    optimizer = make_optimizer(inputs, observers=[collector])
    assert optimizer.run(fmax=1e-3, steps=60), "the batch has to reach the minimum"

    first, last = collector.frames[0], collector.frames[-1]
    assert first.step == 0 and not first.final
    assert last.step == optimizer.nsteps and last.final
    assert first.positions.shape == (6, 3)
    assert first.cell.shape == (2, 3, 3)
    assert first.energy.shape == (2,)
    assert first.max_force_per_config.shape == (2,)

    # the harmonic calculator pulls every atom towards the origin
    torch.testing.assert_close(last.forces, -last.positions)
    # a relaxation towards the origin shrinks the positions and the energy
    assert last.positions.abs().max() < first.positions.abs().max()
    assert last.energy.sum() < first.energy.sum()
    # and every structure ends up flagged converged, none of them starts out that way
    assert last.converged.all()
    assert not first.converged.any()


def test_collected_frames_are_snapshots_not_aliases():
    """The optimizer hands out its live tensors, so a collector has to clone them."""
    collector = FrameCollector(interval=1)
    optimizer = make_optimizer(make_inputs([2]), observers=[collector])
    optimizer.run(fmax=1e-6, steps=6)

    positions = [frame.positions for frame in collector.frames]
    assert not torch.equal(positions[0], positions[-1])


def test_nothing_is_recorded_without_an_observer():
    """No observer means no frame is ever built, so nothing is transferred or written."""
    optimizer = make_optimizer(make_inputs([2]))
    assert optimizer.observers == []

    frames = []
    optimizer._frame = lambda final: frames.append(final)
    optimizer.run(fmax=1e-3, steps=5)
    assert frames == []


# ---------------------------------------------------------------------- log writer


def test_log_writer_writes_one_line_per_recorded_step(tmp_path):
    path = tmp_path / "relax.log"
    optimizer = make_optimizer(make_inputs([2]), logfile=str(path), log_interval=1)
    optimizer.run(fmax=1e-6, steps=6)
    optimizer.close()

    lines = path.read_text().splitlines()
    header, entries = lines[0], lines[1:]
    assert header.split() == ["Step", "Time", "fmax"]
    assert len(entries) == optimizer.nsteps + 1
    assert all(line.startswith("BatchwiseLBFGS:") for line in entries)

    # the logged fmax is the largest force in the batch, and it falls as it relaxes
    logged = [float(line.split()[-1]) for line in entries]
    assert logged[-1] < logged[0]


def test_log_writer_honours_its_interval(tmp_path):
    path = tmp_path / "relax.log"
    optimizer = make_optimizer(make_inputs([2]), logfile=str(path), log_interval=3)
    optimizer.run(fmax=1e-6, steps=10)
    optimizer.close()

    steps = [int(line.split()[1]) for line in path.read_text().splitlines()[1:]]
    assert steps[0] == 0 and steps[-1] == optimizer.nsteps
    assert set(steps) == {0, optimizer.nsteps} | {
        step for step in range(optimizer.nsteps + 1) if step % 3 == 0
    }


def test_logfile_none_writes_nothing(tmp_path):
    """``logfile=None`` adds no observer at all, so no file appears."""
    optimizer = make_optimizer(make_inputs([3]), logfile=None)
    assert optimizer.observers == []
    optimizer.run(fmax=1e-3, steps=5)
    optimizer.close()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------- trajectory recorder


def test_trajectory_holds_one_frame_per_step(tmp_path):
    """``trajectory_interval=1`` records every state the batch passed through."""
    path = str(tmp_path / "relax.hdf5")
    optimizer = make_optimizer(
        make_inputs([3, 3]), trajectory=path, trajectory_interval=1
    )
    optimizer.run(fmax=1e-3, steps=5)
    optimizer.close()

    with BatchwiseTrajectoryReader(path) as traj:
        # the initial state plus one frame after each step
        assert traj.n_frames == optimizer.nsteps + 1
        assert traj.n_structures == 2
        assert traj.n_atoms == 3
        assert traj.positions.shape == (traj.n_frames, 2, 3, 3)
        assert list(traj.steps) == list(range(traj.n_frames))
        # a relaxation towards the origin has to shrink the positions monotonically
        assert np.abs(traj.positions[-1]).max() < np.abs(traj.positions[0]).max()
        assert not traj.has_forces


def test_trajectory_interval_zero_keeps_only_the_endpoints(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    optimizer = make_optimizer(make_inputs([2]), trajectory=path, trajectory_interval=0)
    optimizer.run(fmax=1e-3, steps=20)
    optimizer.close()

    with BatchwiseTrajectoryReader(path) as traj:
        assert list(traj.steps) == [0, optimizer.nsteps]


def test_trajectory_can_store_forces_and_energies(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    inputs = make_inputs([2])
    optimizer = make_optimizer(
        inputs, trajectory=path, trajectory_interval=1, store_forces=True
    )
    optimizer.run(fmax=1e-3, steps=10)
    optimizer.close()

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.has_forces
        # the harmonic calculator pulls every atom towards the origin
        np.testing.assert_allclose(
            traj.forces[-1].reshape(-1, 3),
            -traj.positions[-1].reshape(-1, 3),
            atol=1e-5,
        )
        # and the energy falls as the structures relax
        assert traj.energy[-1].sum() < traj.energy[0].sum()
        # every structure ends up flagged converged
        assert traj.converged[-1].all()
        assert not traj.converged[0].any()


def test_trajectory_metadata_comes_from_the_optimizer(tmp_path):
    """``on_start`` creates the file, so its metadata is settled before frame one."""
    path = str(tmp_path / "relax.hdf5")
    optimizer = make_optimizer(make_inputs([2]), trajectory=path, trajectory_interval=0)
    optimizer.run(fmax=0.01, steps=5)
    optimizer.close()

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.file.attrs["optimizer"] == "BatchwiseLBFGS"
        assert traj.file.attrs["fmax"] == pytest.approx(0.01)


def test_no_trajectory_is_written_without_a_path(tmp_path):
    optimizer = make_optimizer(
        make_inputs([2]), trajectory=str(tmp_path / "relax.hdf5")
    )
    optimizer.close()
    assert (
        list(tmp_path.iterdir()) == []
    ), "the file is created by the run, not by close"

    optimizer = make_optimizer(make_inputs([2]), trajectory=None)
    optimizer.run(fmax=1e-3, steps=5)
    optimizer.close()
    assert list(tmp_path.iterdir()) == []


def test_optimizer_is_a_context_manager(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    with make_optimizer(
        make_inputs([2]), trajectory=path, trajectory_interval=1
    ) as optimizer:
        optimizer.run(fmax=1e-3, steps=5)

    with BatchwiseTrajectoryReader(path) as traj:
        assert traj.n_frames == optimizer.nsteps + 1


def test_close_is_idempotent(tmp_path):
    optimizer = make_optimizer(
        make_inputs([2]), trajectory=str(tmp_path / "relax.hdf5"), log_interval=1
    )
    optimizer.run(fmax=1e-3, steps=5)
    optimizer.close()
    optimizer.close()


def test_relaxation_finds_the_analytic_minimum():
    inputs = make_inputs([6, 6])
    optimizer = make_optimizer(inputs)

    assert optimizer.run(fmax=1e-4, steps=100)
    np.testing.assert_allclose(
        inputs[properties.R].numpy(), np.zeros((12, 3)), atol=1e-4
    )


def test_batch_to_atoms_handles_ragged_batches():
    """The helper is the inverse of atoms_to_batch, so it must not assume equal sizes.

    The optimizer itself rejects ragged batches, but the helper is also used on batches
    that never went through it.
    """
    inputs = {
        properties.n_atoms: torch.tensor([2, 3]),
        properties.R: torch.arange(15, dtype=torch.float32).view(5, 3),
        properties.Z: torch.tensor([1, 6, 8, 1, 1]),
        properties.cell: torch.zeros(2, 3, 3),
        properties.pbc: torch.tensor([[True] * 3, [False] * 3]),
    }

    structures = batch_to_atoms(inputs)

    assert [len(s) for s in structures] == [2, 3]
    assert list(structures[0].numbers) == [1, 6]
    assert list(structures[1].numbers) == [8, 1, 1]
    np.testing.assert_allclose(structures[1].positions, np.arange(6, 15).reshape(3, 3))
    assert structures[0].pbc.all() and not structures[1].pbc.any()


def test_batch_to_atoms_keeps_the_full_cell():
    """Off-diagonal cell entries must survive; taking the diagonal was a real bug."""
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.5, 1.0, 5.0]])
    inputs = make_inputs([3], cell=cell)

    np.testing.assert_allclose(batch_to_atoms(inputs)[0].cell[:], cell, atol=1e-5)


def test_atoms_to_batch_round_trips():
    """The two boundary conversions are each other's inverse, ragged batches included."""
    structures = [
        Atoms(
            numbers=[1, 6],
            positions=np.arange(6).reshape(2, 3) * 0.5,
            cell=np.eye(3) * 4,
            pbc=True,
        ),
        Atoms(
            numbers=[8, 1, 1], positions=np.arange(9).reshape(3, 3) * 0.25, pbc=False
        ),
    ]

    recovered = batch_to_atoms(atoms_to_batch(structures, dtype=torch.float64))

    assert [len(s) for s in recovered] == [2, 3]
    for original, structure in zip(structures, recovered):
        assert list(original.numbers) == list(structure.numbers)
        np.testing.assert_allclose(original.positions, structure.positions, atol=1e-6)
        np.testing.assert_allclose(original.cell[:], structure.cell[:], atol=1e-6)
        assert (original.pbc == structure.pbc).all()

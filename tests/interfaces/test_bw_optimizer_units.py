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


def test_logfile_none_disables_logging():
    optimizer = make_optimizer(make_inputs([3]), logfile=None)
    assert optimizer.logfile is None
    optimizer.run(fmax=1e-3, steps=5)


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


def test_trajectory_interval_n_always_includes_first_and_last(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    optimizer = make_optimizer(make_inputs([2]), trajectory=path, trajectory_interval=3)
    optimizer.run(fmax=1e-6, steps=10)
    optimizer.close()

    with BatchwiseTrajectoryReader(path) as traj:
        steps = list(traj.steps)
    assert steps[0] == 0 and steps[-1] == optimizer.nsteps
    assert set(steps) == {0, optimizer.nsteps} | {
        s for s in range(optimizer.nsteps + 1) if s % 3 == 0
    }


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


def test_no_trajectory_is_written_without_a_path(tmp_path):
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

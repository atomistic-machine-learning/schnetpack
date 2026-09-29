"""Behaviour of the ``Relaxer`` that does not need a trained model.

``test_relaxer_vs_ase.py`` checks that a relaxation lands where ase's ``LBFGS`` lands.
The mechanics around it -- what comes back out of the batch, which atoms are allowed
to move, what an invalid batch does, what the observers see -- are cheaper and clearer
to pin down against an analytic potential, which is what this module does.
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from schnetpack import properties
from schnetpack.dynamics import (
    LBFGS,
    Calculator,
    EulerMaruyama,
    ForceFieldCalculator,
    FrameCollector,
    Interval,
    Relaxer,
    Sampler,
    StateConstraint,
    TrajectoryReader,
)
from schnetpack.generative import VP, VelocityParametrization
from schnetpack.interfaces.ase_interface import atoms_to_batch, batch_to_atoms


class HarmonicModel:
    """Every atom is pulled towards the origin by a spring; the minimum is known."""

    def __init__(self, spring_constant: float = 1.0):
        self.spring_constant = spring_constant
        self.calls = 0

    def __call__(self, inputs):
        self.calls += 1
        positions = inputs[properties.R]
        per_atom = 0.5 * self.spring_constant * positions.pow(2).sum(-1)
        n_structures = inputs[properties.n_atoms].shape[0]
        energy = torch.zeros(n_structures, dtype=positions.dtype).index_add(
            0, inputs[properties.idx_m], per_atom
        )
        return {"energy": energy, "forces": -self.spring_constant * positions}


def make_inputs(n_atoms_per_config, cell=None, seed=0, fixed=None):
    """A schnetpack input batch, built without a converter or a neighbor list."""
    n_configs = len(n_atoms_per_config)
    n_total = sum(n_atoms_per_config)
    rng = np.random.default_rng(seed)

    cells = torch.zeros(n_configs, 3, 3, dtype=torch.float32)
    if cell is not None:
        cells[:] = torch.as_tensor(cell, dtype=torch.float32)

    inputs = {
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
    if fixed is not None:
        inputs[properties.fixed_atoms] = torch.tensor(fixed)
    return inputs


def make_relaxer(model=None, **kwargs) -> Relaxer:
    return Relaxer(model if model is not None else HarmonicModel(), **kwargs)


KCAL = 1 / 23.0605480121  # eV per kcal/mol


def harmonic_in_kcal_and_nm() -> ForceFieldCalculator:
    """The default spring of 1 eV/Ang^2, as a model in kcal/mol and nm would give it."""
    return ForceFieldCalculator(
        HarmonicModel(spring_constant=100.0 / KCAL),
        energy_unit="kcal/mol",
        position_unit="nm",
    )


def test_full_cell_survives_the_round_trip():
    """A triclinic cell must come back as it went in, off-diagonal entries included."""
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.5, 1.0, 5.0]])
    result = make_relaxer().relax(make_inputs([4, 4], cell=cell), 5)

    for structure in batch_to_atoms(result.batch):
        np.testing.assert_allclose(structure.cell[:], cell, atol=1e-5)
        assert structure.pbc.all()


def test_no_mask_matches_an_all_free_mask():
    without = make_relaxer().relax(make_inputs([5, 5]), 20, fmax=1e-3)
    explicit = make_relaxer().relax(make_inputs([5, 5], fixed=[False] * 10), 20, 1e-3)

    torch.testing.assert_close(
        without.batch[properties.R], explicit.batch[properties.R]
    )


def test_fixed_atoms_do_not_move():
    fixed = [True, False, False, False, False] * 2
    inputs = make_inputs([5, 5], fixed=fixed)

    relaxed = make_relaxer().relax(inputs, 20, fmax=1e-3).batch

    moved = (relaxed[properties.R] - inputs[properties.R]).abs().max(dim=1).values
    assert torch.equal(moved[torch.tensor(fixed)], torch.zeros(2))
    assert (moved[~torch.tensor(fixed)] > 1e-3).all()


@pytest.mark.parametrize("integrator", [LBFGS(), EulerMaruyama()])
def test_fixed_atoms_do_not_move_whatever_the_step_rule(integrator):
    fixed = [True, False, False] * 2
    inputs = make_inputs([3, 3], fixed=fixed)

    relaxed = make_relaxer(integrator=integrator, step_size=0.5).relax(inputs, 10).batch

    mask = torch.tensor(fixed)
    assert torch.equal(relaxed[properties.R][mask], inputs[properties.R][mask])


def test_fixed_atoms_are_left_out_of_the_convergence_check():
    """A fixed atom far from the minimum must not keep the relaxation running."""
    inputs = make_inputs([2], fixed=[True, False])
    # the fixed atom carries a large force that can never be relaxed away
    inputs[properties.R] = torch.tensor([[50.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    assert make_relaxer().relax(inputs, 50, fmax=0.05).converged.all()


def test_ragged_batches_are_rejected_by_lbfgs():
    with pytest.raises(ValueError, match="same number of atoms"):
        make_relaxer().relax(make_inputs([3, 4]), 5)


def test_ragged_batches_relax_under_steepest_descent():
    """The relaxer itself is per-structure throughout; only LBFGS needs equal sizes."""
    result = make_relaxer(integrator=EulerMaruyama(), step_size=0.5).relax(
        make_inputs([3, 4]), 100, fmax=1e-4
    )
    assert result.converged.all()
    torch.testing.assert_close(
        result.batch[properties.R], torch.zeros(7, 3), atol=1e-4, rtol=0
    )


def test_steepest_descent_steps_by_step_size():
    """Euler on the force field is x <- x + step_size * F."""
    inputs = make_inputs([2])
    result = make_relaxer(integrator=EulerMaruyama(), step_size=0.25).relax(
        inputs, 1, fmax=1e-12
    )
    torch.testing.assert_close(result.batch[properties.R], inputs[properties.R] * 0.75)


def test_the_step_limit_is_reported_as_failure():
    result = make_relaxer().relax(make_inputs([4]), 2, fmax=1e-12)

    assert not result.converged.any()
    assert result.n_steps == 2


def test_converged_structures_do_not_move():
    inputs = make_inputs([3, 3])
    # structure 0 starts at the minimum
    inputs[properties.R][:3] = 0.0

    relaxed = make_relaxer().relax(inputs, 5, fmax=1e-3).batch

    assert torch.equal(relaxed[properties.R][:3], torch.zeros(3, 3))


@pytest.mark.parametrize("integrator", [LBFGS(), EulerMaruyama()])
def test_converged_structures_do_not_move_whatever_the_step_rule(integrator):
    inputs = make_inputs([3, 3])
    # structure 0 starts below fmax, but off the minimum: its force is not zero
    inputs[properties.R][:3] = 1e-4

    relaxed = make_relaxer(integrator=integrator, step_size=0.5).relax(
        inputs, 5, fmax=1e-3
    )

    assert torch.equal(relaxed.batch[properties.R][:3], inputs[properties.R][:3])
    assert not torch.equal(relaxed.batch[properties.R][3:], inputs[properties.R][3:])


def test_the_force_field_is_the_relaxers_forces_without_diffusion():
    relaxer = make_relaxer()
    inputs = make_inputs([3, 3], fixed=[True, False, False] * 2)
    x = inputs[properties.R]

    field = relaxer.force_field(inputs, x)
    t = torch.zeros(x.shape[0])

    torch.testing.assert_close(field.drift(x, t), relaxer._forces(inputs)[0])
    torch.testing.assert_close(
        field.drift(2 * x, t), relaxer._forces({**inputs, properties.R: 2 * x})[0]
    )
    assert torch.equal(field.diffusion(t), torch.zeros_like(t))
    assert torch.equal(field.n_atoms, inputs[properties.n_atoms])
    assert torch.equal(field.idx_m, inputs[properties.idx_m])


def test_relaxation_finds_the_analytic_minimum():
    result = make_relaxer().relax(make_inputs([6, 6]), 100, fmax=1e-4)

    assert result.converged.all()
    np.testing.assert_allclose(
        result.batch[properties.R].numpy(), np.zeros((12, 3)), atol=1e-4
    )
    torch.testing.assert_close(
        result.outputs["energy"], torch.zeros(2), atol=1e-7, rtol=0
    )


def test_a_model_in_other_units_relaxes_the_same_angstrom_batch():
    """A model in kcal/mol and nm takes the same steps on the same batch in Angstrom."""
    inputs = make_inputs([3])

    ev = make_relaxer().relax(inputs, 3, fmax=1e-12)
    other = make_relaxer(harmonic_in_kcal_and_nm()).relax(inputs, 3, fmax=1e-12)

    torch.testing.assert_close(other.batch[properties.R], ev.batch[properties.R])
    torch.testing.assert_close(other.outputs["forces"], ev.outputs["forces"])
    torch.testing.assert_close(other.outputs["energy"], ev.outputs["energy"])


def test_the_input_batch_is_left_alone_and_nothing_leaks_into_the_output():
    inputs = make_inputs([3, 3])
    before = {k: v.clone() for k, v in inputs.items()}

    relaxed = make_relaxer().relax(inputs, 10).batch

    for key, value in before.items():
        assert torch.equal(inputs[key], value)
    assert set(relaxed) == set(inputs)


def test_a_relaxer_can_be_reused():
    relaxer = make_relaxer()
    first = relaxer.relax(make_inputs([3, 3]), 30, fmax=1e-4)
    second = relaxer.relax(make_inputs([3, 3]), 30, fmax=1e-4)
    fresh = make_relaxer().relax(make_inputs([3, 3]), 30, fmax=1e-4)

    assert first.n_steps == second.n_steps == fresh.n_steps
    torch.testing.assert_close(second.batch[properties.R], fresh.batch[properties.R])


def test_one_model_call_per_step():
    model = HarmonicModel()
    result = make_relaxer(model).relax(make_inputs([3, 3]), 50, fmax=1e-4)

    # one for the initial forces, one per step taken
    assert model.calls == result.n_steps + 1


def test_a_given_calculator_gets_its_cache_turned_on():
    calculator = ForceFieldCalculator(HarmonicModel())
    make_relaxer(calculator)
    assert calculator.cache_last


def test_a_bare_model_is_taken_as_a_force_field_in_ev_and_angstrom():
    calculator = make_relaxer().calculator
    assert isinstance(calculator, ForceFieldCalculator)
    assert calculator.enable_grad
    assert calculator.energy_conversion == calculator.position_conversion == 1.0


def test_a_calculator_without_units_is_refused():
    with pytest.raises(TypeError, match="ForceFieldCalculator"):
        make_relaxer(Calculator(HarmonicModel()))


def test_a_model_without_forces_is_reported():
    with pytest.raises(KeyError, match="forces"):
        make_relaxer(lambda batch: {"energy": torch.zeros(1)}).relax(
            make_inputs([2]), 3
        )


def test_denoise_returns_the_relaxed_batch():
    inputs = make_inputs([3, 3])
    assert torch.equal(
        make_relaxer().denoise(inputs, 10)[properties.R],
        make_relaxer().relax(inputs, 10).batch[properties.R],
    )


def test_sample_relaxes_prior_draws():
    class NoisyStructures:
        """A prior in the sense of Dynamics.sample: n -> a batch of n structures."""

        def sample(self, n):
            return make_inputs([4] * n, seed=n)

    relaxed = make_relaxer(prior=NoisyStructures()).sample(2, 100)
    # sample() relaxes to the default fmax of 0.05 eV/Ang, i.e. |x| < 0.05 here
    assert relaxed[properties.R].norm(dim=-1).max() < 0.05


def test_constraints_run_around_every_step():
    class Recorder(StateConstraint):
        def __init__(self):
            self.calls = []

        def before_step(self, batch, step, n_steps, dynamics):
            self.calls.append(("before", step))
            return batch

        def after_step(self, batch, step, n_steps, dynamics):
            self.calls.append(("after", step))
            return batch

    recorder = Recorder()
    result = make_relaxer(constraints=[recorder]).relax(make_inputs([2]), 3, 1e-12)

    assert result.n_steps == 3
    assert recorder.calls == [
        ("before", 0),
        ("after", 1),
        ("before", 1),
        ("after", 2),
        ("before", 2),
        ("after", 3),
    ]


def test_the_sampler_refuses_a_per_structure_step_rule():
    with pytest.raises(ValueError, match="Relaxer"):
        Sampler(lambda b: b, VP(), VelocityParametrization(), LBFGS())


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
    assert interval.due(7, final=True)


def test_observers_only_see_the_steps_they_asked_for():
    every = FrameCollector(interval=1)
    endpoints = FrameCollector(interval=0)
    result = make_relaxer(observers=[every, endpoints]).relax(
        make_inputs([2]), 6, fmax=1e-6
    )

    assert every.steps == list(range(result.n_steps + 1))
    assert endpoints.steps == [0, result.n_steps]


def test_a_frame_describes_the_state_it_was_taken_from():
    collector = FrameCollector(interval=1)
    result = make_relaxer(observers=[collector]).relax(
        make_inputs([3, 3]), 60, fmax=1e-3
    )
    assert result.converged.all(), "the batch has to reach the minimum"

    first, last = collector.frames[0], collector.frames[-1]
    assert first.step == 0 and not first.final
    assert last.step == result.n_steps and last.final
    assert first.positions.shape == (6, 3)
    assert first.batch[properties.cell].shape == (2, 3, 3)
    assert first.energy.shape == (2,)
    assert first.max_force_per_config.shape == (2,)

    # the harmonic model pulls every atom towards the origin
    torch.testing.assert_close(last.forces, -last.positions)
    assert last.positions.abs().max() < first.positions.abs().max()
    assert last.energy.sum() < first.energy.sum()
    assert last.converged.all()
    assert not first.converged.any()


def test_collected_frames_are_snapshots_not_aliases():
    collector = FrameCollector(interval=1)
    make_relaxer(observers=[collector]).relax(make_inputs([2]), 6, fmax=1e-6)

    positions = [frame.positions for frame in collector.frames]
    assert not torch.equal(positions[0], positions[-1])


def test_nothing_is_built_without_an_observer():
    relaxer = make_relaxer()
    assert relaxer.observers == []

    built = []
    relaxer._frame = lambda *args: built.append(args) or (lambda: None)
    relaxer.relax(make_inputs([2]), 5, fmax=1e-3)
    # the builders are made, but none is ever called
    assert all(len(args) == 7 for args in built)


def test_negative_step_limit_is_rejected():
    with pytest.raises(ValueError, match="n_steps"):
        make_relaxer().relax(make_inputs([2]), -1)


def test_zero_step_limit_evaluates_the_start():
    batch = make_inputs([2])
    result = make_relaxer().relax(batch, 0, fmax=1e-6)
    assert result.n_steps == 0
    assert torch.equal(result.batch[properties.position], batch[properties.position])


# ---------------------------------------------------------------------- log writer


def test_log_writer_writes_one_line_per_recorded_step(tmp_path):
    path = tmp_path / "relax.log"
    result = make_relaxer(logfile=str(path), log_interval=1).relax(
        make_inputs([2]), 6, fmax=1e-6
    )

    lines = path.read_text().splitlines()
    header, entries = lines[0], lines[1:]
    assert header.split() == ["Step", "Time", "fmax"]
    assert len(entries) == result.n_steps + 1
    assert all(line.startswith("LBFGS:") for line in entries)

    logged = [float(line.split()[-1]) for line in entries]
    assert logged[-1] < logged[0]


def test_log_writer_honours_its_interval(tmp_path):
    path = tmp_path / "relax.log"
    result = make_relaxer(logfile=str(path), log_interval=3).relax(
        make_inputs([2]), 10, fmax=1e-6
    )

    steps = [int(line.split()[1]) for line in path.read_text().splitlines()[1:]]
    assert steps[0] == 0 and steps[-1] == result.n_steps
    assert set(steps) == {0, result.n_steps} | {
        step for step in range(result.n_steps + 1) if step % 3 == 0
    }


def test_logfile_none_writes_nothing(tmp_path):
    relaxer = make_relaxer(logfile=None)
    assert relaxer.observers == []
    relaxer.relax(make_inputs([3]), 5, fmax=1e-3)
    assert list(tmp_path.iterdir()) == []


def test_log_file_is_closed_after_the_run(tmp_path):
    relaxer = make_relaxer(logfile=str(tmp_path / "relax.log"))
    relaxer.relax(make_inputs([2]), 3)
    assert relaxer.observers[0].file is None


# --------------------------------------------------------------- trajectory recorder


def test_trajectory_holds_one_frame_per_step(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    result = make_relaxer(trajectory=path, trajectory_interval=1).relax(
        make_inputs([3, 3]), 5, fmax=1e-3
    )

    with TrajectoryReader(path) as traj:
        assert traj.n_frames == result.n_steps + 1
        assert traj.n_structures == 2
        assert traj.n_atoms.tolist() == [3, 3]
        assert traj.positions.shape == (traj.n_frames, 6, 3)
        assert list(traj.steps) == list(range(traj.n_frames))
        assert np.abs(traj.positions[-1]).max() < np.abs(traj.positions[0]).max()
        assert not traj.has_forces


def test_trajectory_interval_zero_keeps_only_the_endpoints(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    result = make_relaxer(trajectory=path, trajectory_interval=0).relax(
        make_inputs([2]), 20, fmax=1e-3
    )

    with TrajectoryReader(path) as traj:
        assert list(traj.steps) == [0, result.n_steps]


def test_trajectory_can_store_forces_and_energies(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    make_relaxer(trajectory=path, trajectory_interval=1, store_forces=True).relax(
        make_inputs([2]), 10, fmax=1e-3
    )

    with TrajectoryReader(path) as traj:
        assert traj.has_forces
        np.testing.assert_allclose(traj.forces[-1], -traj.positions[-1], atol=1e-5)
        assert traj.energy[-1].sum() < traj.energy[0].sum()
        assert traj.converged[-1].all()
        assert not traj.converged[0].any()


def test_trajectory_metadata_comes_from_the_relaxer(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    make_relaxer(trajectory=path, trajectory_interval=0).relax(
        make_inputs([2]), 5, fmax=0.01
    )

    with TrajectoryReader(path) as traj:
        assert traj.file.attrs["driver"] == "Relaxer"
        assert traj.file.attrs["integrator"] == "LBFGS"
        assert traj.file.attrs["fmax"] == pytest.approx(0.01)


def test_the_trajectory_is_in_angstrom_whatever_the_models_units(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    inputs = make_inputs([3])
    make_relaxer(
        harmonic_in_kcal_and_nm(),
        trajectory=path,
        trajectory_interval=1,
        store_forces=True,
    ).relax(inputs, 3, fmax=1e-12)

    with TrajectoryReader(path) as traj:
        np.testing.assert_allclose(traj.positions[0], inputs[properties.R].numpy())
        # a spring of 1 eV/Ang^2: the forces in eV/Ang are minus the positions in Ang
        np.testing.assert_allclose(
            traj.forces[:], -traj.positions[:], rtol=1e-5, atol=1e-6
        )


def test_no_trajectory_is_written_before_a_run(tmp_path):
    make_relaxer(trajectory=str(tmp_path / "relax.hdf5"))
    assert list(tmp_path.iterdir()) == [], "the file is created by the run"


def test_the_trajectory_is_closed_even_if_the_run_fails(tmp_path):
    path = str(tmp_path / "relax.hdf5")
    relaxer = make_relaxer(trajectory=path)

    class Boom(StateConstraint):
        def after_step(self, batch, step, n_steps, dynamics):
            raise RuntimeError("boom")

    relaxer.constraints.append(Boom())
    with pytest.raises(RuntimeError, match="boom"):
        relaxer.relax(make_inputs([2]), 5, fmax=1e-12)
    assert relaxer.observers[0].writer is None
    with TrajectoryReader(path) as traj:
        assert list(traj.steps) == [0]


# ------------------------------------------------------------- ase boundary helpers


def test_batch_to_atoms_handles_ragged_batches():
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
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.5, 1.0, 5.0]])
    inputs = make_inputs([3], cell=cell)

    np.testing.assert_allclose(batch_to_atoms(inputs)[0].cell[:], cell, atol=1e-5)


def test_atoms_to_batch_round_trips():
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

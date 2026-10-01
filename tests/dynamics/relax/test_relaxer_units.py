"""Behaviour of the time-free drivers (``LBFGS``, ``Langevin``) that does not need a trained model.

``test_relaxer_vs_ase.py`` checks that a relaxation lands where ase's ``LBFGS`` lands.
The mechanics around it -- what comes back out of the batch, which atoms are allowed
to move, and what an invalid batch does -- are cheaper and clearer
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
    ForceCalculator,
    Langevin,
    StateConstraint,
)
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


def make_relaxer(model=None, **kwargs) -> LBFGS:
    return LBFGS(model if model is not None else HarmonicModel(), **kwargs)


def make_descent(model=None, step_size=0.5, **kwargs) -> Langevin:
    """Steepest descent: Langevin at kT = 0."""
    model = model if model is not None else HarmonicModel()
    return Langevin(model, step_size=step_size, **kwargs)


class StepCounter(StateConstraint):
    """Counts the steps a relaxation takes; ``denoise`` returns only the batch."""

    def __init__(self):
        self.steps = 0

    def after_step(self, batch, step, n_steps, dynamics):
        self.steps = step
        return batch


def relax_counting(relaxer, batch, n_steps):
    """The relaxed batch and the number of steps it took."""
    counter = StepCounter()
    relaxer.constraints.append(counter)
    try:
        return relaxer.denoise(batch, n_steps), counter.steps
    finally:
        relaxer.constraints.remove(counter)


KCAL = 1 / 23.0605480121  # eV per kcal/mol


def harmonic_in_kcal_and_nm() -> ForceCalculator:
    """The default spring of 1 eV/Ang^2, as a model in kcal/mol and nm would give it."""
    return ForceCalculator(
        HarmonicModel(spring_constant=100.0 / KCAL),
        energy_unit="kcal/mol",
        position_unit="nm",
    )


def test_full_cell_survives_the_round_trip():
    """A triclinic cell must come back as it went in, off-diagonal entries included."""
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.5, 1.0, 5.0]])
    relaxed = make_relaxer().denoise(make_inputs([4, 4], cell=cell), 5)

    for structure in batch_to_atoms(relaxed):
        np.testing.assert_allclose(structure.cell[:], cell, atol=1e-5)
        assert structure.pbc.all()


def test_no_mask_matches_an_all_free_mask():
    without = make_relaxer(fmax=1e-3).denoise(make_inputs([5, 5]), 20)
    explicit = make_relaxer(fmax=1e-3).denoise(
        make_inputs([5, 5], fixed=[False] * 10), 20
    )

    torch.testing.assert_close(without[properties.R], explicit[properties.R])


def test_fixed_atoms_do_not_move():
    fixed = [True, False, False, False, False] * 2
    inputs = make_inputs([5, 5], fixed=fixed)

    relaxed = make_relaxer(fmax=1e-3).denoise(inputs, 20)

    moved = (relaxed[properties.R] - inputs[properties.R]).abs().max(dim=1).values
    assert torch.equal(moved[torch.tensor(fixed)], torch.zeros(2))
    assert (moved[~torch.tensor(fixed)] > 1e-3).all()


#: the two step rules on the same harmonic surface, built with the given keywords
DRIVERS = pytest.mark.parametrize(
    "build", [make_relaxer, make_descent], ids=["LBFGS", "steepest-descent"]
)


@DRIVERS
def test_fixed_atoms_do_not_move_whatever_the_step_rule(build):
    fixed = [True, False, False] * 2
    inputs = make_inputs([3, 3], fixed=fixed)

    relaxed = build().denoise(inputs, 10)

    mask = torch.tensor(fixed)
    assert torch.equal(relaxed[properties.R][mask], inputs[properties.R][mask])


def test_fixed_atoms_are_left_out_of_the_convergence_check():
    """A fixed atom far from the minimum must not keep the relaxation running."""
    inputs = make_inputs([2], fixed=[True, False])
    # the fixed atom carries a large force that can never be relaxed away
    inputs[properties.R] = torch.tensor([[50.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    relaxed, steps = relax_counting(make_relaxer(fmax=0.05), inputs, 50)

    assert steps < 50
    assert relaxed[properties.R][1].norm() < 0.05


def test_ragged_batches_relax_under_lbfgs():
    relaxed = make_relaxer(fmax=1e-4).denoise(make_inputs([3, 4, 1]), 100)

    torch.testing.assert_close(
        relaxed[properties.R], torch.zeros(8, 3), atol=1e-4, rtol=0
    )


def test_a_ragged_batch_relaxes_each_structure_as_it_would_alone():
    """The per-structure reductions of LBFGS must not mix up the structures."""
    sizes = [2, 5, 3]
    inputs = make_inputs(sizes)

    relaxed = make_relaxer(fmax=1e-12).denoise(inputs, 4)

    for positions, alone in zip(
        inputs[properties.R].split(sizes), relaxed[properties.R].split(sizes)
    ):
        single = make_inputs([len(positions)])
        single[properties.R] = positions
        expected = make_relaxer(fmax=1e-12).denoise(single, 4)[properties.R]
        torch.testing.assert_close(alone, expected)


def test_lbfgs_limits_the_step_per_structure_in_a_ragged_batch():
    """A structure far from the minimum must not shrink its neighbour's step."""
    inputs = make_inputs([3, 2])
    inputs[properties.R][:3] *= 100.0  # its first step H0 * F is far above maxstep
    inputs[properties.R][3:] *= 0.1  # its first step is far below
    relaxer = make_relaxer(maxstep=0.2, fmax=1e-12)

    relaxed = relaxer.denoise(inputs, 1)

    step = relaxed[properties.R] - inputs[properties.R]
    assert step[:3].norm(dim=-1).max() == pytest.approx(0.2, rel=1e-5)
    # the harmonic force is -x, so the unscaled first step is -H0 x
    torch.testing.assert_close(step[3:], -relaxer.H0 * inputs[properties.R][3:])


def test_ragged_batches_relax_under_steepest_descent():
    """The loop itself is per-structure throughout."""
    relaxed = make_descent(fmax=1e-4).denoise(make_inputs([3, 4]), 100)
    torch.testing.assert_close(
        relaxed[properties.R], torch.zeros(7, 3), atol=1e-4, rtol=0
    )


def test_steepest_descent_steps_by_step_size():
    """Gradient descent on the force field is x <- x + step_size * F."""
    inputs = make_inputs([2])
    relaxed = make_descent(step_size=0.25, fmax=1e-12).denoise(inputs, 1)
    torch.testing.assert_close(relaxed[properties.R], inputs[properties.R] * 0.75)


def test_the_step_limit_stops_an_unconverged_run():
    relaxed, steps = relax_counting(make_relaxer(fmax=1e-12), make_inputs([4]), 2)

    assert steps == 2
    assert relaxed[properties.R].norm(dim=-1).max() > 1e-12


def test_converged_structures_do_not_move():
    inputs = make_inputs([3, 3])
    # structure 0 starts at the minimum
    inputs[properties.R][:3] = 0.0

    relaxed = make_relaxer(fmax=1e-3).denoise(inputs, 5)

    assert torch.equal(relaxed[properties.R][:3], torch.zeros(3, 3))


@DRIVERS
def test_converged_structures_do_not_move_whatever_the_step_rule(build):
    inputs = make_inputs([3, 3])
    # structure 0 starts below fmax, but off the minimum: its force is not zero
    inputs[properties.R][:3] = 1e-4

    relaxed = build(fmax=1e-3).denoise(inputs, 5)

    assert torch.equal(relaxed[properties.R][:3], inputs[properties.R][:3])
    assert not torch.equal(relaxed[properties.R][3:], inputs[properties.R][3:])


def test_the_forces_are_zero_on_fixed_atoms():
    inputs = make_inputs([3, 3], fixed=[True, False, False] * 2)
    forces = make_relaxer().forces(inputs)

    mask = inputs[properties.fixed_atoms]
    assert torch.equal(forces[mask], torch.zeros(2, 3))
    torch.testing.assert_close(forces[~mask], -inputs[properties.R][~mask])


def test_relaxation_finds_the_analytic_minimum():
    relaxed = make_relaxer(fmax=1e-4).denoise(make_inputs([6, 6]), 100)

    np.testing.assert_allclose(
        relaxed[properties.R].numpy(), np.zeros((12, 3)), atol=1e-4
    )


def test_a_model_in_other_units_relaxes_the_same_angstrom_batch():
    """A model in kcal/mol and nm takes the same steps on the same batch in Angstrom."""
    inputs = make_inputs([3])

    ev = make_relaxer(fmax=1e-12).denoise(inputs, 3)
    other = make_relaxer(harmonic_in_kcal_and_nm(), fmax=1e-12).denoise(inputs, 3)

    torch.testing.assert_close(other[properties.R], ev[properties.R])


def test_the_input_batch_is_left_alone_and_nothing_leaks_into_the_output():
    inputs = make_inputs([3, 3])
    before = {k: v.clone() for k, v in inputs.items()}

    relaxed = make_relaxer().denoise(inputs, 10)

    for key, value in before.items():
        assert torch.equal(inputs[key], value)
    assert set(relaxed) == set(inputs)


def test_a_relaxer_can_be_reused():
    relaxer = make_relaxer(fmax=1e-4)
    first, first_steps = relax_counting(relaxer, make_inputs([3, 3]), 30)
    second, second_steps = relax_counting(relaxer, make_inputs([3, 3]), 30)
    fresh, fresh_steps = relax_counting(
        make_relaxer(fmax=1e-4), make_inputs([3, 3]), 30
    )

    assert first_steps == second_steps == fresh_steps
    torch.testing.assert_close(second[properties.R], fresh[properties.R])


def test_one_model_call_per_step():
    model = HarmonicModel()
    _, steps = relax_counting(make_relaxer(model, fmax=1e-4), make_inputs([3, 3]), 50)

    # one for the initial forces, one per step taken
    assert model.calls == steps + 1


def test_a_given_calculator_gets_its_cache_turned_on():
    calculator = ForceCalculator(HarmonicModel())
    make_relaxer(calculator)
    assert calculator.cache_last


def test_a_bare_model_is_taken_as_a_force_field_in_ev_and_angstrom():
    calculator = make_relaxer().calculator
    assert isinstance(calculator, ForceCalculator)
    assert calculator.enable_grad
    assert calculator.energy_conversion == calculator.position_conversion == 1.0


def test_a_calculator_without_units_is_refused():
    with pytest.raises(TypeError, match="ForceCalculator"):
        make_relaxer(Calculator(HarmonicModel()))


def test_a_model_without_forces_is_reported():
    with pytest.raises(KeyError, match="forces"):
        make_relaxer(lambda batch: {"energy": torch.zeros(1)}).denoise(
            make_inputs([2]), 3
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
    make_relaxer(constraints=[recorder], fmax=1e-12).denoise(make_inputs([2]), 3)

    assert recorder.calls == [
        ("before", 0),
        ("after", 1),
        ("before", 1),
        ("after", 2),
        ("before", 2),
        ("after", 3),
    ]


def test_negative_step_limit_is_rejected():
    with pytest.raises(ValueError, match="n_steps"):
        make_relaxer().denoise(make_inputs([2]), -1)


def test_zero_step_limit_evaluates_the_start():
    batch = make_inputs([2])
    relaxed, steps = relax_counting(make_relaxer(fmax=1e-6), batch, 0)
    assert steps == 0
    assert torch.equal(relaxed[properties.position], batch[properties.position])


# ------------------------------------------------------------- langevin, pseudo-forces


def test_langevin_samples_the_boltzmann_density():
    """On the unit spring exp(-|x|^2 / 2kT) has variance kT per coordinate."""
    torch.manual_seed(0)
    kT = 0.1
    inputs = make_inputs([2000])
    sampled = make_descent(step_size=0.01, kT=kT).denoise(inputs, 1000)
    variance = sampled[properties.R].double().var().item()
    assert variance == pytest.approx(kT, rel=0.05)


def test_langevin_validates_its_temperature():
    with pytest.raises(ValueError, match="fmax"):
        make_descent(kT=0.1, fmax=0.05)
    with pytest.raises(ValueError, match="non-negative"):
        make_descent(kT=-1.0)


def test_lbfgs_on_a_pseudo_force_starts_with_the_gpff_jump():
    """alpha = 2 by default, so the first step x + F/2 lands on x0."""

    def gpff(batch):
        return {"prediction": -2.0 * batch[properties.R]}  # x0 = 0

    inputs = make_inputs([3, 2])
    inputs[properties.R] *= 0.1
    relaxer = make_relaxer(ForceCalculator(gpff, kind="pseudo"), maxstep=1.0)
    assert relaxer.H0 == pytest.approx(0.5)

    relaxed = relaxer.denoise(inputs, 1)
    torch.testing.assert_close(relaxed[properties.R], torch.zeros(5, 3))


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

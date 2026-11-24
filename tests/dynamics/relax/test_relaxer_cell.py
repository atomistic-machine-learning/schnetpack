"""Cell steps of the batch-wise relaxers (``relax_cell=True``) against ase's ``LBFGS`` on
``FrechetCellFilter``.

The crystals are small EMT metals, rattled and strained. The comparison is on the whole
path, as for the clusters in ``test_relaxer_vs_ase.py``: same variables, same step rule
and same stop test give the same number of steps and the same end point.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from schnetpack import dynamics, properties
from schnetpack.dynamics import ForceCalculator
from schnetpack.interfaces.ase_interface import atoms_to_batch, batch_to_atoms

from .test_relaxer_units import HarmonicModel, relax_counting
from .test_relaxer_vs_ase import EMTModel

FMAX = 0.01
MAX_STEPS = 200


def make_crystals() -> list[Atoms]:
    """Three fcc metals of 4, 8 and 12 atoms, each rattled and strained on its own."""
    crystals = [
        bulk("Cu", cubic=True),
        bulk("Al", cubic=True) * (2, 1, 1),
        bulk("Ni", cubic=True) * (1, 3, 1),
    ]
    for seed, crystal in enumerate(crystals):
        strain = np.random.default_rng(seed).normal(scale=0.03, size=(3, 3))
        crystal.set_cell(crystal.cell @ (np.eye(3) + strain), scale_atoms=True)
        crystal.rattle(stdev=0.05, seed=seed)
    return crystals


def relax_with_ase(crystals: list[Atoms], fixed=None) -> tuple[list[Atoms], list[int]]:
    """Each crystal relaxed alone by ase's ``LBFGS`` on ``FrechetCellFilter``."""
    relaxed, steps = [], []
    for crystal in deepcopy(crystals):
        crystal.calc = EMT()
        if fixed is not None:
            crystal.set_constraint(FixAtoms(fixed))
        optimizer = LBFGS(FrechetCellFilter(crystal), logfile=None)
        optimizer.run(fmax=FMAX, steps=MAX_STEPS)
        relaxed.append(crystal)
        steps.append(optimizer.nsteps)
    return relaxed, steps


def relaxer(**kwargs) -> dynamics.LBFGS:
    calculator = ForceCalculator(EMTModel(stress=True), stress_key="stress")
    return dynamics.LBFGS(calculator, fmax=FMAX, relax_cell=True, **kwargs)


def to_batch(crystals: list[Atoms]) -> dict[str, torch.Tensor]:
    return atoms_to_batch(deepcopy(crystals), dtype=torch.float64)


def test_the_models_stress_is_the_strain_derivative_of_its_energy():
    """The sign and unit the cell steps rely on: stress = dE/dstrain / volume."""
    crystal = make_crystals()[1]
    model = EMTModel(stress=True)
    stress = model(to_batch([crystal]))["stress"][0]

    def energy(strain: np.ndarray) -> float:
        strained = crystal.copy()
        strained.set_cell(crystal.cell @ (np.eye(3) + strain), scale_atoms=True)
        return float(model(to_batch([strained]))["energy"])

    numerical = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            strain = np.zeros((3, 3))
            # a symmetric strain, so no rotation enters the derivative
            strain[i, j] += 0.5e-5
            strain[j, i] += 0.5e-5
            numerical[i, j] = (energy(strain) - energy(-strain)) / 2e-5
    np.testing.assert_allclose(
        stress.numpy(), numerical / crystal.get_volume(), atol=1e-7
    )


def test_a_batch_of_crystals_relaxes_as_ase_relaxes_each_with_its_cell_filter():
    """Crystals of different sizes relax together, cell and positions, as ase's
    ``LBFGS`` on ``FrechetCellFilter`` relaxes each one alone."""
    crystals = make_crystals()
    alone, steps_alone = relax_with_ase(crystals)

    relaxed, steps = relax_counting(relaxer(), to_batch(crystals), MAX_STEPS)

    assert max(steps_alone) < MAX_STEPS
    # the batch runs until its slowest structure is relaxed
    assert steps == max(steps_alone)
    for start, reference, in_batch in zip(crystals, alone, batch_to_atoms(relaxed)):
        # the cell has to travel, or the comparison below shows nothing
        assert np.abs(in_batch.cell.array - start.cell.array).max() > 0.05
        np.testing.assert_allclose(in_batch.cell.array, reference.cell.array, atol=1e-8)
        np.testing.assert_allclose(in_batch.positions, reference.positions, atol=1e-8)
        in_batch.calc = EMT()
        assert abs(in_batch.get_stress()).max() < FMAX
        assert in_batch.get_potential_energy() == pytest.approx(
            reference.get_potential_energy(), abs=1e-10
        )


def test_a_crystal_relaxes_the_same_alone_and_in_a_batch():
    crystals = make_crystals()
    in_batch = batch_to_atoms(relaxer().run(to_batch(crystals), MAX_STEPS))
    for crystal, together in zip(crystals, in_batch):
        (alone,) = batch_to_atoms(relaxer().run(to_batch([crystal]), MAX_STEPS))
        np.testing.assert_allclose(alone.cell.array, together.cell.array, atol=1e-10)
        np.testing.assert_allclose(alone.positions, together.positions, atol=1e-10)


def test_a_converged_crystal_is_held_cell_and_positions():
    """A relaxed crystal beside a strained one keeps its cell and its positions while
    the other one relaxes."""
    crystals = make_crystals()
    (done,), _ = relax_with_ase(crystals[:1])
    done.calc = None
    start = to_batch([done, crystals[1]])

    relaxed, steps = relax_counting(relaxer(), start, MAX_STEPS)

    assert 0 < steps < MAX_STEPS
    n_done = len(done)
    assert torch.allclose(
        relaxed[properties.cell][0], start[properties.cell][0], atol=1e-13
    )
    assert torch.allclose(
        relaxed[properties.R][:n_done], start[properties.R][:n_done], atol=1e-13
    )
    assert not torch.allclose(
        relaxed[properties.cell][1], start[properties.cell][1], atol=1e-3
    )


def test_fixed_atoms_keep_their_fractional_coordinates_as_under_ase():
    """A fixed atom follows the cell at fixed fractional coordinates, which is what
    ``FixAtoms`` does under ase's cell filter."""
    crystals = make_crystals()[1:2]
    alone, _ = relax_with_ase(crystals, fixed=[0, 3])

    start = to_batch(crystals)
    fixed = torch.zeros(len(crystals[0]), dtype=torch.bool)
    fixed[[0, 3]] = True
    start[properties.fixed_atoms] = fixed
    (relaxed,) = batch_to_atoms(relaxer().run(start, MAX_STEPS))

    np.testing.assert_allclose(relaxed.cell.array, alone[0].cell.array, atol=1e-8)
    np.testing.assert_allclose(relaxed.positions, alone[0].positions, atol=1e-8)
    np.testing.assert_allclose(
        relaxed.get_scaled_positions(wrap=False)[[0, 3]],
        crystals[0].get_scaled_positions(wrap=False)[[0, 3]],
        atol=1e-12,
    )


def test_gradient_descent_relaxes_the_cell_too():
    """Any step rule gets the cell rows: steepest descent lowers the energy and moves
    the cell."""
    crystals = make_crystals()[:1]
    model = EMTModel(stress=True)
    start = to_batch(crystals)
    descent = dynamics.GradientDescent(
        ForceCalculator(model, stress_key="stress"), step_size=0.01, relax_cell=True
    )
    relaxed = descent.run(start, 20)

    assert float(model(relaxed)["energy"]) < float(model(start)["energy"]) - 1e-3
    assert not torch.allclose(
        relaxed[properties.cell], start[properties.cell], atol=1e-3
    )


def test_cell_steps_need_a_stress():
    with pytest.raises(ValueError, match="stress_key"):
        dynamics.LBFGS(HarmonicModel(), relax_cell=True)

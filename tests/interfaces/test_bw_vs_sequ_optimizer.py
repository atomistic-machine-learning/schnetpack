"""Compare ``ASEBatchwiseLBFGS`` against a sequential loop over ase ``LBFGS``.

The efficiency assertions are on the number of model forward passes rather than on wall
clock time. Both calculators count them (``n_fwd_iterations``), and the count is
independent of the machine the tests run on. Timings are measured and reported, but
never asserted on.
"""

import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read
from ase.optimize import LBFGS

import schnetpack as spk
from schnetpack.interfaces.ase_interface import AtomsConverter, SpkCalculator
from schnetpack.interfaces.batchwise_optimization import (
    ASEBatchwiseLBFGS,
    BatchwiseCalculator,
)
from schnetpack.utils.compatibility import load_model


TESTDATA = os.path.join(os.path.dirname(__file__), "..", "testdata")
MODEL_PATH = os.path.join(TESTDATA, "md_ethanol.model")
STRUCTURE_PATH = os.path.join(TESTDATA, "ethanol_conformers.xyz")

DEVICE = torch.device("cpu")
SEED = 0
# a multiple of the three conformers, so each one seeds an equal share of the batch
N_STRUCTURES = 9
# Displacement applied to the conformers to create the batch. The conformers are
# themselves already relaxed (fmax ~1e-5), so without it there would be nothing to
# optimize. Kept moderate on purpose: at 0.1 Ang the sequential optimizer occasionally
# walks into a hole in the model and diverges, which would make it a useless reference.
NOISE = 0.05
FMAX = 0.01
MAX_STEPS = 200

# Both optimizers must be scored by the same units. BatchwiseCalculator defaults to eV
# while SpkCalculator defaults to kcal/mol, so neither default is left in place.
ENERGY_UNIT = "kcal/mol"
POSITION_UNIT = "Ang"


@dataclass
class RelaxationResult:
    """Outcome of relaxing a batch of structures, however it was relaxed."""

    atoms: List[Atoms]
    steps: List[int]  # optimizer steps, per structure for the sequential run
    forward_passes: int  # model calls, the hardware independent cost measure
    wall_time: float


def _neighbor_list():
    model = load_model(MODEL_PATH, device=DEVICE)
    return spk.transform.MatScipyNeighborList(cutoff=model.representation.cutoff.item())


def _spk_calculator():
    return SpkCalculator(
        model_file=MODEL_PATH,
        neighbor_list=_neighbor_list(),
        device=DEVICE,
        energy_unit=ENERGY_UNIT,
        position_unit=POSITION_UNIT,
    )


def _relax_batchwise(atoms_list: List[Atoms]) -> RelaxationResult:
    """Relax a batch of structures in parallel with ``ASEBatchwiseLBFGS``."""
    converter = AtomsConverter(neighbor_list=_neighbor_list(), device=DEVICE)
    calculator = BatchwiseCalculator(
        model=MODEL_PATH,
        atoms_converter=converter,
        device=DEVICE,
        energy_unit=ENERGY_UNIT,
        position_unit=POSITION_UNIT,
    )
    inputs = converter(deepcopy(atoms_list))

    n_atoms = len(atoms_list[0])
    optimizer = ASEBatchwiseLBFGS(
        calculator=calculator,
        inputs=inputs,
        logfile=None,
        trajectory=None,
        # rebuilding ase Atoms on every step would dominate the measured time
        log_every_step=False,
        # an all-False mask means "no atom is fixed"; passing None instead would add a
        # leading dimension when the mask is used to index the positions
        fixed_atoms_mask=[False] * (n_atoms * len(atoms_list)),
        # the constructor defaults to cuda
        device=DEVICE,
    )

    start = time.perf_counter()
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)
    wall_time = time.perf_counter() - start

    relaxed, _ = optimizer.get_relaxation_results()
    return RelaxationResult(
        atoms=relaxed,
        steps=[optimizer.nsteps],
        forward_passes=calculator.n_fwd_iterations,
        wall_time=wall_time,
    )


@pytest.fixture(scope="module")
def shared_calculator():
    """One calculator used to score every relaxed structure, whatever produced it."""
    return _spk_calculator()


@pytest.fixture(scope="module")
def initial_structures():
    """A batch seeded by cycling the ethanol conformers, each one displaced.

    Using several conformers rather than one reference geometry means the batch
    converges into more than one basin, so the comparison against the sequential run
    has to reproduce a pattern of distinct minima instead of a single collapsed value.
    """
    conformers = read(STRUCTURE_PATH, index=":")
    rng = np.random.default_rng(SEED)
    structures = []
    for idx in range(N_STRUCTURES):
        noisy = conformers[idx % len(conformers)].copy()
        noisy.positions += rng.normal(scale=NOISE, size=noisy.positions.shape)
        structures.append(noisy)
    return structures


@pytest.fixture(scope="module")
def sequential_result(initial_structures):
    """Relax the structures one at a time, the way ase would normally be used."""
    calculator = _spk_calculator()

    atoms, steps, forward_passes = [], [], []
    start = time.perf_counter()
    # LBFGS relaxes in place, so the shared initial structures must not be handed over
    for structure in deepcopy(initial_structures):
        before = calculator.n_fwd_iterations
        structure.calc = calculator
        optimizer = LBFGS(structure, logfile=None)
        optimizer.run(fmax=FMAX, steps=MAX_STEPS)

        atoms.append(structure)
        steps.append(optimizer.nsteps)
        forward_passes.append(calculator.n_fwd_iterations - before)
    wall_time = time.perf_counter() - start

    result = RelaxationResult(
        atoms=atoms,
        steps=steps,
        forward_passes=sum(forward_passes),
        wall_time=wall_time,
    )
    # kept separately: the per structure split is what the efficiency metrics need
    result.forward_passes_per_structure = forward_passes
    return result


@pytest.fixture(scope="module")
def batchwise_result(initial_structures):
    return _relax_batchwise(initial_structures)


@pytest.fixture(scope="module")
def single_structure_atoms(initial_structures):
    """Every structure relaxed on its own, but still through the batch-wise optimizer."""
    return [_relax_batchwise([structure]).atoms[0] for structure in initial_structures]


def evaluate(atoms_list: List[Atoms], calculator):
    """Energies and max force norms, all from the same calculator."""
    energies, fmax = [], []
    for structure in atoms_list:
        structure = structure.copy()
        structure.calc = calculator
        energies.append(structure.get_potential_energy())
        fmax.append(np.sqrt((structure.get_forces() ** 2).sum(axis=1).max()))
    return np.array(energies), np.array(fmax)


def test_both_optimizers_converge(
    sequential_result, batchwise_result, shared_calculator
):
    """Neither run may be compared against a reference that never converged."""
    assert max(sequential_result.steps) < MAX_STEPS, (
        "sequential relaxation hit the step limit, so it is not a valid reference: "
        f"steps={sequential_result.steps}"
    )
    assert batchwise_result.steps[0] < MAX_STEPS

    _, fmax_seq = evaluate(sequential_result.atoms, shared_calculator)
    _, fmax_batch = evaluate(batchwise_result.atoms, shared_calculator)

    assert fmax_seq.max() <= FMAX, f"sequential did not reach fmax: {fmax_seq}"
    assert fmax_batch.max() <= FMAX, f"batch-wise did not reach fmax: {fmax_batch}"


def test_batchwise_reaches_same_minima(
    sequential_result, batchwise_result, shared_calculator
):
    """Both optimizers must land in the same basin.

    Energy is the right quantity to compare: it is invariant under the rigid body
    motion and the relabelling of identical atoms that a relaxation may introduce, so
    no structural alignment is needed. The conformers seeding this batch settle into
    two basins 2.7 meV apart, comfortably above the tolerance below, so a structure
    ending up in the wrong one would still be caught.
    """
    energies_seq, _ = evaluate(sequential_result.atoms, shared_calculator)
    energies_batch, _ = evaluate(batchwise_result.atoms, shared_calculator)

    np.testing.assert_allclose(energies_batch, energies_seq, atol=1e-3)


def test_batchwise_needs_no_more_iterations_than_slowest_structure(
    sequential_result, batchwise_result
):
    """The core invariant: batching must not degrade convergence.

    The batch runs until its slowest member converges, so it should need about as many
    forward passes as that member needs on its own. Needing more means the batching
    itself is slowing the optimization down.
    """
    slowest = max(sequential_result.forward_passes_per_structure)
    assert batchwise_result.forward_passes <= slowest + 2, (
        f"batch-wise needed {batchwise_result.forward_passes} forward passes, but the "
        f"slowest structure alone needs only {slowest}"
    )


def test_batchwise_uses_fewer_forward_passes(
    sequential_result, batchwise_result, record_property, benchmark_report
):
    """The batch must actually amortize the model calls across the structures."""
    steps = sequential_result.steps
    load_balance = float(np.mean(steps) / max(steps))
    speedup = sequential_result.wall_time / batchwise_result.wall_time
    saving = sequential_result.forward_passes / batchwise_result.forward_passes

    record_property("forward_passes_batchwise", batchwise_result.forward_passes)
    record_property("forward_passes_sequential", sequential_result.forward_passes)
    record_property("speedup", speedup)
    record_property("load_balance_efficiency", load_balance)
    record_property("batching_efficiency", speedup / load_balance)

    benchmark_report(
        f"{N_STRUCTURES} ethanol structures relaxed to fmax={FMAX} on {DEVICE}"
    )
    benchmark_report(
        f"  forward passes   batch-wise {batchwise_result.forward_passes:4d} "
        f"| sequential {sequential_result.forward_passes:4d}  ({saving:.1f}x fewer)"
    )
    benchmark_report(
        f"  wall time        batch-wise {batchwise_result.wall_time:6.2f} s "
        f"| sequential {sequential_result.wall_time:6.2f} s"
    )
    benchmark_report(f"  speedup      S   {speedup:.2f}")
    benchmark_report(
        f"  load balance L   {load_balance:.3f}  (mean/max sequential steps)"
    )
    benchmark_report(f"  batching     B   {speedup / load_balance:.2f}  (= S / L)")

    assert 3 * batchwise_result.forward_passes <= sequential_result.forward_passes, (
        f"batch-wise used {batchwise_result.forward_passes} forward passes against "
        f"{sequential_result.forward_passes} sequential ones, less than the expected "
        f"3x saving for a batch of {N_STRUCTURES}"
    )


def test_batch_size_invariance(batchwise_result, single_structure_atoms):
    """Relaxing a structure alone or inside a batch must give the same result.

    This exercises the batching machinery itself, without ase in the picture: the
    per structure inverse Hessian, the mask that freezes already converged structures,
    and the per structure step rescaling. Any leakage between batch members shows up
    here. The tolerance covers float32 position accumulation.
    """
    for idx, (in_batch, alone) in enumerate(
        zip(batchwise_result.atoms, single_structure_atoms)
    ):
        deviation = np.abs(in_batch.get_positions() - alone.get_positions()).max()
        assert deviation < 1e-4, (
            f"structure {idx} relaxed differently inside a batch of {N_STRUCTURES} "
            f"than on its own: max deviation {deviation:.2e} Ang"
        )

"""Compare ``BatchwiseLBFGS`` against a sequential loop over ase ``LBFGS``.

The comparison is on the outcome of the relaxation only: both optimizers must land in
the same minima, and a structure must relax the same way alone as inside a batch. Wall
clock timing lives in ``test_bw_vs_sequ_benchmark.py``, which measures it with
pytest-benchmark.
"""

import os
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
from schnetpack import properties
from schnetpack.interfaces.ase_interface import (
    AtomsConverter,
    SpkCalculator,
    batch_to_atoms,
)
from schnetpack.interfaces.batchwise_optimization import (
    BatchwiseCalculator,
    BatchwiseLBFGS,
)
from schnetpack.utils.compatibility import load_model

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "testdata")
MODEL_PATH = os.path.join(TESTDATA, "md_ethanol.model")
STRUCTURE_PATH = os.path.join(TESTDATA, "ethanol_conformers.xyz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
N_STRUCTURES = 9  # multiple of number of conformers
NOISE = 0.05  # added to atomic positions
FMAX = 0.01
MAX_STEPS = 200
CUTOFF_SKIN = 0.3  # Ang, reuse the neighbor list while atoms move less than half
ENERGY_UNIT = "kcal/mol"
POSITION_UNIT = "Ang"


@dataclass
class RelaxationResult:
    """Outcome of relaxing a batch of structures, however it was relaxed."""

    atoms: List[Atoms]
    steps: List[int]  # optimizer steps, per structure for the sequential run


def _neighbor_list(cutoff_skin: float = 0.0):
    """The neighbor list both optimizers get.

    ``cutoff_skin > 0`` wraps it in a ``SkinNeighborList``, which lets a relaxation
    reuse the previous list while no atom has moved more than half the skin. That is
    what makes the batch-wise path worth using, so the batch-wise optimizer gets it
    and the sequential ase reference does not (ase rebuilds per structure anyway).
    """
    model = load_model(MODEL_PATH, device=DEVICE)
    neighbor_list = spk.transform.MatScipyNeighborList(
        cutoff=model.representation.cutoff.item()
    )
    if cutoff_skin > 0.0:
        neighbor_list = spk.transform.SkinNeighborList(
            neighbor_list=neighbor_list, cutoff_skin=cutoff_skin
        )
    return neighbor_list


def spk_calculator():
    return SpkCalculator(
        model=MODEL_PATH,
        neighbor_list=_neighbor_list(),
        device=DEVICE,
        energy_unit=ENERGY_UNIT,
        position_unit=POSITION_UNIT,
    )


def make_structures(n_structures: int = N_STRUCTURES) -> List[Atoms]:
    """A batch seeded by cycling the ethanol conformers,
    and adding noise to atomic positions.
    """
    conformers = read(STRUCTURE_PATH, index=":")
    rng = np.random.default_rng(SEED)
    structures = []
    for idx in range(n_structures):
        noisy = conformers[idx % len(conformers)].copy()
        noisy.positions += rng.normal(scale=NOISE, size=noisy.positions.shape)
        structures.append(noisy)
    return structures


def build_batchwise_optimizer(atoms_list: List[Atoms]) -> BatchwiseLBFGS:
    """Everything needed to relax a batch, short of actually running it.

    Kept separate from the run so the benchmark can time only the relaxation.
    """
    converter = AtomsConverter(
        neighbor_list=_neighbor_list(cutoff_skin=CUTOFF_SKIN), device=DEVICE
    )
    calculator = BatchwiseCalculator(
        model=MODEL_PATH,
        atoms_converter=converter,
        device=DEVICE,
        energy_unit=ENERGY_UNIT,
        position_unit=POSITION_UNIT,
    )
    inputs = converter(deepcopy(atoms_list))

    n_atoms = len(atoms_list[0])
    return BatchwiseLBFGS(
        calculator=calculator,
        inputs=inputs,
        logfile=None,
        fixed_atoms_mask=[False] * (n_atoms * len(atoms_list)),
    )


def relax_batchwise(atoms_list: List[Atoms]) -> RelaxationResult:
    """Relax a batch of structures in parallel with ``BatchwiseLBFGS``."""
    optimizer = build_batchwise_optimizer(atoms_list)
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)

    # the optimizer hands back tensors; ase structures are a boundary conversion
    relaxed, _ = optimizer.get_relaxation_results()
    return RelaxationResult(atoms=batch_to_atoms(relaxed), steps=[optimizer.nsteps])


def relax_sequential(atoms_list: List[Atoms], calculator) -> RelaxationResult:
    """Relax the structures one at a time, the way ase would normally be used."""
    atoms, steps = [], []
    # LBFGS relaxes in place, so the caller's structures must not be handed over
    for structure in deepcopy(atoms_list):
        structure.calc = calculator
        optimizer = LBFGS(structure, logfile=None)
        optimizer.run(fmax=FMAX, steps=MAX_STEPS)

        atoms.append(structure)
        steps.append(optimizer.nsteps)

    return RelaxationResult(atoms=atoms, steps=steps)


@pytest.fixture(scope="module")
def shared_calculator():
    """One calculator used to score every relaxed structure, whatever produced it."""
    return spk_calculator()


@pytest.fixture(scope="module")
def initial_structures():
    return make_structures()


@pytest.fixture(scope="module")
def sequential_result(initial_structures):
    return relax_sequential(initial_structures, spk_calculator())


@pytest.fixture(scope="module")
def batchwise_result(initial_structures):
    return relax_batchwise(initial_structures)


@pytest.fixture(scope="module")
def single_structure_atoms(initial_structures):
    """Every structure relaxed on its own, but still through the batch-wise optimizer."""
    return [relax_batchwise([structure]).atoms[0] for structure in initial_structures]


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


def test_batch_size_invariance(batchwise_result, single_structure_atoms):
    """Relaxing a structure alone or inside a batch must give the same result."""
    for idx, (in_batch, alone) in enumerate(
        zip(batchwise_result.atoms, single_structure_atoms)
    ):
        deviation = np.abs(in_batch.get_positions() - alone.get_positions()).max()
        assert deviation < 1e-4, (
            f"structure {idx} relaxed differently inside a batch of {N_STRUCTURES} "
            f"than on its own: max deviation {deviation:.2e} Ang"
        )


@pytest.mark.parametrize("trajectory_interval", [0, 1])
def test_forces_are_computed_once_per_step(
    initial_structures, tmp_path, trajectory_interval
):
    """The convergence check, the frame written from it, and the step share one call.

    The calculator caches its results and decides whether they are still valid from
    the identity and mutation counter of the position tensor. If that check ever goes
    wrong in the conservative direction, relaxations silently cost twice as much.
    Writing a frame on every step must not cost a second call either, which is why
    ``trajectory_interval=1`` is covered here too.
    """
    optimizer = build_batchwise_optimizer(initial_structures[:3])
    optimizer.trajectory = str(tmp_path / "relax.hdf5")
    optimizer.trajectory_interval = trajectory_interval
    calculate = optimizer.calculator.calculate
    calls = []

    def counting_calculate(inputs):
        calls.append(None)
        return calculate(inputs)

    optimizer.calculator.calculate = counting_calculate
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)
    optimizer.close()

    # one for the initial forces, one per step taken
    assert len(calls) == optimizer.nsteps + 1


def test_cached_forces_are_dropped_when_the_positions_move(initial_structures):
    """The other direction: a structure that changed must not return stale forces."""
    optimizer = build_batchwise_optimizer(initial_structures[:2])
    calculator, inputs = optimizer.calculator, optimizer.inputs

    before = calculator.get_forces(inputs).clone()
    assert torch.equal(calculator.get_forces(inputs), before), "cache should have hit"

    inputs[properties.R] += 0.1
    assert not torch.equal(calculator.get_forces(inputs), before)

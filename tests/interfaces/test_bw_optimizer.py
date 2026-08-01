"""Compare ``ASEBatchwiseLBFGS`` against a sequential loop over ase ``LBFGS``.

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
from schnetpack.interfaces.ase_interface import AtomsConverter, SpkCalculator
from schnetpack.interfaces.batchwise_optimization import (
    ASEBatchwiseLBFGS,
    BatchwiseCalculator,
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
ENERGY_UNIT = "kcal/mol"
POSITION_UNIT = "Ang"


@dataclass
class RelaxationResult:
    """Outcome of relaxing a batch of structures, however it was relaxed."""

    atoms: List[Atoms]
    steps: List[int]  # optimizer steps, per structure for the sequential run


def _neighbor_list():
    model = load_model(MODEL_PATH, device=DEVICE)
    return spk.transform.MatScipyNeighborList(cutoff=model.representation.cutoff.item())


def spk_calculator():
    return SpkCalculator(
        model_file=MODEL_PATH,
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


def build_batchwise_optimizer(atoms_list: List[Atoms]) -> ASEBatchwiseLBFGS:
    """Everything needed to relax a batch, short of actually running it.

    Kept separate from the run so the benchmark can time only the relaxation.
    """
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
    return ASEBatchwiseLBFGS(
        calculator=calculator,
        inputs=inputs,
        logfile=None,
        trajectory=None,
        log_every_step=False,
        fixed_atoms_mask=[False] * (n_atoms * len(atoms_list)),
        device=DEVICE,
    )


def relax_batchwise(atoms_list: List[Atoms]) -> RelaxationResult:
    """Relax a batch of structures in parallel with ``ASEBatchwiseLBFGS``."""
    optimizer = build_batchwise_optimizer(atoms_list)
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)

    relaxed, _ = optimizer.get_relaxation_results()
    return RelaxationResult(atoms=relaxed, steps=[optimizer.nsteps])


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

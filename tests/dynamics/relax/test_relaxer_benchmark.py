"""Wall clock scaling of the batch-wise ``dynamics.LBFGS`` against sequential ase ``LBFGS``.

Nothing here asserts on time, since wall clock thresholds are machine dependent.

These are deselected by default. Run them with::
    pytest tests/dynamics/relax -m benchmark_sweep --benchmark-group-by=param --benchmark-time-unit=ms
"""

import pytest

from schnetpack.interfaces.ase_interface import atoms_to_batch

from .test_relaxer_units import relax_counting
from .test_relaxer_vs_ase import (
    DEVICE,
    MAX_STEPS,
    build_relaxer,
    make_structures,
    relax_sequential,
    spk_calculator,
)

# ensure that batch size appears in ascending order
N_VALUES = [pytest.param(n, id=f"{n:02d}") for n in (1, 3, 9, 18)]
ROUNDS = 3


@pytest.mark.benchmark_sweep
@pytest.mark.parametrize("n_structures", N_VALUES)
def test_batchwise_relaxation(benchmark, n_structures):
    structures = make_structures(n_structures)
    steps = []

    def setup():
        # a fresh relaxer and batch for every round, left out of the timing
        relaxer = build_relaxer()
        inputs = relaxer.calculator.prepare(atoms_to_batch(structures, device=DEVICE))
        return (relaxer, inputs), {}

    def run(relaxer, inputs):
        steps.append(relax_counting(relaxer, inputs, MAX_STEPS)[1])

    benchmark.pedantic(run, setup=setup, rounds=ROUNDS, iterations=1)

    # a run that is fast because it never converged is not a faster run
    assert steps[-1] < MAX_STEPS


@pytest.mark.benchmark_sweep
@pytest.mark.parametrize("n_structures", N_VALUES)
def test_sequential_relaxation(benchmark, n_structures):
    structures = make_structures(n_structures)
    results = []

    def setup():
        # Sequential relaxation copies the structures itself.
        # Only the calculator needs renewing.
        return (structures, spk_calculator()), {}

    def run(atoms_list, calculator):
        results.append(relax_sequential(atoms_list, calculator))

    benchmark.pedantic(run, setup=setup, rounds=ROUNDS, iterations=1)

    assert max(results[-1].steps) < MAX_STEPS

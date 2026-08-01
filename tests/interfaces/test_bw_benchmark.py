"""Wall clock scaling of ``ASEBatchwiseLBFGS`` against sequential ase ``LBFGS``.

Nothing here asserts on time, since wall clock thresholds are machine dependent.

These are deselected by default. Run them with::
    pytest tests/interfaces -m benchmark_sweep --benchmark-group-by=param --benchmark-time-unit=ms
"""

import pytest

from .test_bw_optimizer import (
    FMAX,
    MAX_STEPS,
    build_batchwise_optimizer,
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
    built = []

    def setup():
        # Ensure fresh optimizer for each for every round.
        # Re-run before every round and left out of the timing.
        optimizer = build_batchwise_optimizer(structures)
        built.append(optimizer)
        return (optimizer,), {}

    benchmark.pedantic(
        lambda optimizer: optimizer.run(fmax=FMAX, steps=MAX_STEPS),
        setup=setup,
        rounds=ROUNDS,
        iterations=1,
    )

    # a run that is fast because it never converged is not a faster run
    assert built[-1].nsteps < MAX_STEPS


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

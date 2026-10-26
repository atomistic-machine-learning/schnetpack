import pickle

import ase.build
import pytest
import torch

import schnetpack as spk
import schnetpack.properties as properties
from schnetpack.objectives import ModelOutput

device = "cuda" if torch.cuda.is_available() else "cpu"

_compile_supported = pytest.mark.skipif(
    not spk.model.compile_supported(),
    reason="torch < 2.14 traces wrong gradients of a force loss",
)


def requires_compile(test):
    """Compiling takes seconds per graph, so these tests run only with -m slow."""
    return pytest.mark.slow(_compile_supported(test))


CUTOFF = 3.0
REPRESENTATIONS = [spk.model.SchNet, spk.model.PaiNN, spk.model.SO3net]


def potential(representation, stress=False):
    kwargs = dict(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=CUTOFF),
        cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
    )
    if representation is spk.model.SO3net:
        kwargs["lmax"] = 2
    input_modules = [spk.model.Strain()] if stress else []
    return spk.model.NeuralNetworkPotential(
        representation=representation(**kwargs),
        input_modules=input_modules + [spk.model.PairwiseDistances()],
        output_modules=[
            spk.model.Atomwise(n_in=16, output_key=properties.energy),
            spk.model.Forces(calc_stress=stress),
        ],
        input_dtype_str="float64",
    ).to(device, torch.float64)


def molecules(n_atoms, seed=0):
    """All pairs within each molecule of a batch, random positions."""
    g = torch.Generator().manual_seed(seed)
    idx_i, idx_j, idx_m, start = [], [], [], 0
    for m, n in enumerate(n_atoms):
        i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        keep = i != j
        idx_i.append(i[keep] + start)
        idx_j.append(j[keep] + start)
        idx_m.append(torch.full((n,), m))
        start += n
    idx_i = torch.cat(idx_i)
    batch = {
        properties.Z: torch.randint(1, 10, (start,), generator=g),
        properties.R: torch.rand(start, 3, generator=g, dtype=torch.float64) * 3,
        properties.idx_i: idx_i,
        properties.idx_j: torch.cat(idx_j),
        properties.offsets: torch.zeros(len(idx_i), 3, dtype=torch.float64),
        properties.idx_m: torch.cat(idx_m),
        properties.n_atoms: torch.tensor(n_atoms),
    }
    return {k: v.to(device) for k, v in batch.items()}


def crystals(repeats, seed=0):
    """Rattled silicon cells, pairs across the periodic boundaries."""
    structures = []
    for k, repeat in enumerate(repeats):
        atoms = ase.build.bulk("Si", "diamond", a=5.43).repeat(repeat)
        atoms.rattle(0.1, seed=seed + k)
        structures.append(atoms)
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=spk.transform.ASENeighborList(cutoff=CUTOFF),
        device=device,
        dtype=torch.float64,
    )
    return converter(structures)


def loss_grads(fn, model, batch, keys):
    out = fn(dict(batch))
    loss = sum(out[k].pow(2).sum() for k in keys)
    return torch.autograd.grad(loss, list(model.parameters()))


def assert_matches_eager(model, compiled, batches, keys):
    model.train()
    for batch in batches:
        for a, b in zip(
            loss_grads(compiled, model, batch, keys),
            loss_grads(model, model, batch, keys),
        ):
            torch.testing.assert_close(a, b)

    model.eval()
    for batch in batches:
        expected = model(dict(batch))
        out = compiled(dict(batch))
        assert out[properties.forces].grad_fn is None
        for k in keys:
            torch.testing.assert_close(out[k], expected[k])


@requires_compile
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_compiled_potential_matches_eager(representation):
    torch.manual_seed(0)
    model = potential(representation)
    compiled = spk.model.CompiledPotential(model)
    batches = [molecules((5, 7)), molecules((4, 6, 8), seed=1), molecules((3, 9), 2)]

    assert_matches_eager(
        model, compiled, batches, (properties.energy, properties.forces)
    )
    # one graph per mode, reused across batch sizes
    assert len(compiled._graphs) == 2


@requires_compile
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_compiled_stress_of_periodic_cells_matches_eager(representation):
    torch.manual_seed(0)
    model = potential(representation, stress=True)
    compiled = spk.model.CompiledPotential(model)
    batches = [
        crystals([(1, 1, 1), (1, 1, 2)]),
        crystals([(1, 2, 2), (1, 1, 1), (2, 1, 1)], seed=1),
    ]

    assert_matches_eager(
        model,
        compiled,
        batches,
        (properties.energy, properties.forces, properties.stress),
    )
    assert len(compiled._graphs) == 2


@requires_compile
def test_single_molecule_batch_gets_own_graph():
    """A graph traced on one molecule must not be reused for several, or vice versa."""
    torch.manual_seed(0)
    model = potential(spk.model.SchNet).eval()
    compiled = spk.model.CompiledPotential(model)
    batches = [molecules((9,)), molecules((5, 7), 1), molecules((6,), 2)]

    for batch in batches:
        expected = model(dict(batch))
        out = compiled(dict(batch))
        for k in (properties.energy, properties.forces):
            torch.testing.assert_close(out[k], expected[k])

    assert len(compiled._graphs) == 2


def atomistic_task(compile):
    torch.manual_seed(0)
    return spk.lightning.AtomisticTask(
        model=potential(spk.model.SchNet),
        outputs=[
            ModelOutput(name=properties.energy, loss_fn=torch.nn.MSELoss()),
            ModelOutput(name=properties.forces, loss_fn=torch.nn.MSELoss()),
        ],
        optimizer_args={"lr": 1e-4},
        compile=compile,
    )


def labelled(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_molecules, n_atoms = len(batch[properties.n_atoms]), len(batch[properties.Z])
    labels = {
        properties.energy: torch.randn(n_molecules, generator=g, dtype=torch.float64),
        properties.forces: torch.randn(n_atoms, 3, generator=g, dtype=torch.float64),
    }
    return {**batch, **{k: v.to(device) for k, v in labels.items()}}


@requires_compile
def test_task_trains_the_compiled_model_like_eager():
    compiled, eager = atomistic_task(compile=True), atomistic_task(compile=False)
    assert eager.compiled_model is None
    for batch in [labelled(molecules((5, 7))), labelled(molecules((4, 6, 8), 1), 1)]:
        grads = [
            torch.autograd.grad(
                task.training_step(dict(batch), 0), list(task.model.parameters())
            )
            for task in (compiled, eager)
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b)
    assert len(compiled.compiled_model._graphs) == 1


@requires_compile
def test_task_runs_compiled_model_and_pickles():
    task = atomistic_task(compile=True).eval()
    batch = molecules((5, 7))
    out = task(dict(batch))
    expected = task.model(dict(batch))
    for k in (properties.energy, properties.forces):
        torch.testing.assert_close(out[k], expected[k])

    # spktrain stores the whole task with torch.save
    copy = pickle.loads(pickle.dumps(task))
    assert copy.compiled_model.model is copy.model
    assert copy.compiled_model._graphs == {}
    assert len(task.compiled_model._graphs) == 1


def test_compile_needs_torch_2_14(monkeypatch):
    monkeypatch.setattr("schnetpack.model.compile.compile_supported", lambda: False)
    with pytest.raises(RuntimeError, match="torch >= 2.14"):
        spk.model.CompiledPotential(potential(spk.model.SchNet))
    # the task, spkpredict and the calculators all compile through it
    with pytest.raises(RuntimeError, match="torch >= 2.14"):
        atomistic_task(compile=True)


def test_compile_needs_an_atomistic_model():
    with pytest.raises(TypeError, match="AtomisticModel"):
        spk.model.CompiledPotential(torch.nn.Linear(2, 2))
    with pytest.raises(TypeError, match="AtomisticModel"):
        spk.dynamics.Calculator(lambda batch: batch, compile=True)


@requires_compile
def test_ase_calculator_runs_the_compiled_model():
    torch.manual_seed(0)
    model = potential(spk.model.SchNet)
    calculators = [
        spk.interfaces.SpkCalculator(
            model,
            neighbor_list=spk.transform.ASENeighborList(cutoff=CUTOFF),
            device=device,
            dtype=torch.float64,
            compile_model=compile_model,
        )
        for compile_model in (True, False)
    ]
    atoms = ase.build.molecule("CH3CH2OH")
    for step in range(3):
        atoms.rattle(0.2, seed=step)
        out, expected = (
            calculator.get_forces(atoms.copy()) for calculator in calculators
        )
        torch.testing.assert_close(torch.tensor(out), torch.tensor(expected))
    # a moving molecule with changing neighbors reuses the graph
    assert len(calculators[0].compiled_model._graphs) == 1
    assert calculators[1].compiled_model is None


@requires_compile
def test_md_calculator_runs_the_compiled_model(tmp_path):
    torch.manual_seed(0)
    torch.save(potential(spk.model.SchNet).cpu(), tmp_path / "model")
    forces = []
    for compile_model in (True, False):
        calculator = spk.md.calculators.SchNetPackCalculator(
            model_file=str(tmp_path / "model"),
            force_key=properties.forces,
            energy_unit="eV",
            position_unit="Angstrom",
            neighbor_list=spk.md.neighborlist_md.NeighborListMD(
                cutoff=CUTOFF, cutoff_shell=0.5, base_nbl=spk.transform.ASENeighborList
            ),
            energy_key=properties.energy,
            required_properties=[properties.energy, properties.forces],
            compile_model=compile_model,
        ).to(device)
        system = spk.md.System()
        system.load_molecules(ase.build.molecule("CH3CH2OH"), n_replicas=2)
        system.to(device)
        calculator.calculate(system)
        forces.append(system.forces)
    torch.testing.assert_close(*forces)

    with pytest.raises(spk.md.calculators.MDCalculatorError):
        spk.md.calculators.SchNetPackCalculator(
            model_file=str(tmp_path / "model"),
            force_key=properties.forces,
            energy_unit="eV",
            position_unit="Angstrom",
            neighbor_list=None,
            script_model=True,
            compile_model=True,
        )


@requires_compile
def test_dynamics_calculator_compiles_atomistic_models():
    torch.manual_seed(0)
    model = potential(spk.model.SchNet).eval()
    batch = molecules((5, 7))
    out = spk.dynamics.Calculator(model, compile=True)(batch)
    expected = spk.dynamics.Calculator(model, enable_grad=True)(batch)
    for k in (properties.energy, properties.forces):
        torch.testing.assert_close(out[k], expected[k])

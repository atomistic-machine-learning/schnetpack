"""The LAMMPS interface: LammpsPotential, the graph the pair style runs, against the
model it comes from, and the AOTInductor packages deploy_lammps writes. The domain
test checks the contract the pair style relies on under MPI, without LAMMPS: the
energies, forces and virials of domains with ghost atoms add up to those of the
periodic system. The C++ test builds the part of the pair style that loads and runs a
package (interfaces/lammps/schnetpack_model.cpp) against this torch and compares it to
Python."""

import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import ase.build
import numpy as np
import pytest
import torch

import schnetpack as spk
from schnetpack import properties
from schnetpack.cli import deploy
from schnetpack.interfaces.lammps_interface import (
    LAMMPS_INPUTS,
    LAMMPS_OUTPUTS,
    LammpsPotential,
    _example,
    deploy_lammps,
)

CUTOFF = 3.0
REPRESENTATIONS = [spk.model.SchNet, spk.model.PaiNN, spk.model.SO3net]

PAIR_STYLE = Path(__file__).parents[2] / "interfaces" / "lammps"

requires_compile = pytest.mark.skipif(
    not spk.model.compile_supported(),
    reason="the deploy recipe is validated on torch >= 2.14",
)


def potential(representation, seed=0, output_modules=None, postprocessors=None):
    torch.manual_seed(seed)
    kwargs = dict(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=10, cutoff=CUTOFF),
        cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
    )
    if representation is spk.model.SO3net:
        kwargs["lmax"] = 2
    if postprocessors is None:
        offsets = spk.transform.AddOffsets(
            properties.energy,
            add_mean=True,
            add_atomrefs=True,
            atomrefs=torch.randn(100),
            property_mean=torch.tensor([-1.5]),
        )
        postprocessors = [spk.transform.CastTo64(), offsets]
    return spk.model.NeuralNetworkPotential(
        representation=representation(**kwargs),
        input_modules=[spk.model.Strain(), spk.model.PairwiseDistances()],
        output_modules=output_modules
        or [
            spk.model.Atomwise(n_in=16, output_key=properties.energy),
            spk.model.Forces(calc_stress=True),
        ],
        postprocessors=postprocessors,
        input_dtype_str="float64",
    ).double()


def silicon(repeat=3, seed=0):
    atoms = ase.build.bulk("Si", "diamond", a=5.43).repeat(repeat)
    atoms.rattle(0.1, seed=seed)
    return atoms


def periodic(atoms):
    """The batch of ``atoms`` with periodic shifts, as the model reads it."""
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=spk.transform.ASENeighborList(cutoff=CUTOFF),
        dtype=torch.float64,
    )
    return converter(atoms)


def flat(batch, local=None):
    n_atoms = batch[properties.Z].shape[0]
    return dict(
        positions=batch[properties.R],
        atomic_numbers=batch[properties.Z],
        idx_i=batch[properties.idx_i],
        idx_j=batch[properties.idx_j],
        offsets=batch[properties.offsets],
        local=torch.ones(n_atoms, dtype=torch.float64) if local is None else local,
    )


def domains(atoms, receptive_field, n_domains):
    """
    Split ``atoms`` into slabs along the first cell vector and give each the ghost
    atoms LAMMPS would: every periodic image within ``receptive_field`` of a local
    atom, at its image position, with edges between all atoms within the cutoff and
    no shifts. Yields the inputs and, per atom, the index of the atom it images.
    """
    cell = np.array(atoms.cell)
    scaled = atoms.get_scaled_positions() % 1.0
    images = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    for d in range(n_domains):
        owned = np.floor(scaled[:, 0] * n_domains) == d
        local_pos = scaled[owned] @ cell
        nodes, owner, is_local = list(local_pos), list(np.nonzero(owned)[0]), []
        is_local = [1.0] * len(nodes)
        for shift in images:
            for i in range(len(atoms)):
                if owned[i] and not shift.any():
                    continue
                pos = (scaled[i] + shift) @ cell
                if np.linalg.norm(local_pos - pos, axis=1).min() <= receptive_field:
                    nodes.append(pos)
                    owner.append(i)
                    is_local.append(0.0)
        nodes = np.array(nodes)
        dist = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
        idx_i, idx_j = np.nonzero((dist < CUTOFF) & (dist > 0))
        yield (
            dict(
                positions=torch.tensor(nodes),
                atomic_numbers=torch.tensor(atoms.numbers[owner]),
                idx_i=torch.tensor(idx_i),
                idx_j=torch.tensor(idx_j),
                offsets=torch.zeros(len(idx_i), 3, dtype=torch.float64),
                local=torch.tensor(is_local, dtype=torch.float64),
            ),
            torch.tensor(owner),
        )


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_potential_matches_model(representation):
    model = potential(representation).eval()
    atoms = silicon(repeat=2)
    batch = periodic(atoms)
    expected = model({k: v.clone() for k, v in batch.items()})

    energy, energy_atom, forces, virial = LammpsPotential(model)(**flat(batch))

    torch.testing.assert_close(energy, expected[properties.energy][0])
    torch.testing.assert_close(energy_atom.sum(), energy)
    torch.testing.assert_close(forces, expected[properties.forces])
    stress = expected[properties.stress][0]
    torch.testing.assert_close(virial, -stress * atoms.get_volume())


def test_ghost_atoms_add_nothing_but_forces():
    model = potential(spk.model.SchNet).eval()
    batch = periodic(silicon(repeat=2))
    local = torch.ones(batch[properties.Z].shape[0], dtype=torch.float64)
    local[::2] = 0.0

    energy, energy_atom, forces, _ = LammpsPotential(model)(**flat(batch, local))

    assert (energy_atom[::2] == 0).all()
    torch.testing.assert_close(energy, energy_atom.sum())
    assert forces[::2].abs().max() > 0


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_domains_with_ghost_atoms_add_up_to_the_periodic_system(representation):
    lammps = LammpsPotential(potential(representation).eval())
    atoms = silicon(repeat=3)
    energy, _, forces, virial = lammps(**flat(periodic(atoms)))

    total = torch.zeros(())
    total_forces = torch.zeros_like(forces)
    total_virial = torch.zeros_like(virial)
    for inputs, owner in domains(atoms, lammps.receptive_field, n_domains=3):
        e, _, f, w = lammps(**inputs)
        total = total + e
        # as LAMMPS' reverse communication adds the forces on ghost atoms
        total_forces.index_add_(0, owner, f)
        total_virial = total_virial + w

    torch.testing.assert_close(total, energy)
    torch.testing.assert_close(total_forces, forces)
    torch.testing.assert_close(total_virial, virial)


def test_potential_leaves_the_model_unchanged():
    model = potential(spk.model.PaiNN)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    LammpsPotential(model).float()

    assert all(
        v.dtype == torch.float64
        for v in model.state_dict().values()
        if v.is_floating_point()
    )
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(output_modules=[spk.model.Atomwise(n_in=16)]), "one Forces"),
        (
            dict(
                output_modules=[
                    spk.model.Atomwise(
                        n_in=16, output_key=properties.energy, aggregation_mode="avg"
                    ),
                    spk.model.Forces(),
                ]
            ),
            "must sum",
        ),
        (
            dict(
                output_modules=[
                    spk.model.Atomwise(n_in=16, output_key=properties.energy),
                    spk.model.Atomwise(n_in=16, output_key=properties.energy),
                    spk.model.Forces(),
                ]
            ),
            "found 2",
        ),
        (
            dict(
                postprocessors=[
                    spk.transform.AddOffsets(properties.energy, is_extensive=False)
                ]
            ),
            "intensive",
        ),
        (
            dict(postprocessors=[spk.transform.ScaleProperty(properties.energy)]),
            "ScaleProperty",
        ),
    ],
)
def test_potential_rejects_what_lammps_cannot_run(kwargs, match):
    model = potential(spk.model.SchNet, **kwargs)

    with pytest.raises(ValueError, match=match):
        LammpsPotential(model)


def test_potential_needs_a_neural_network_potential():
    with pytest.raises(TypeError, match="NeuralNetworkPotential"):
        LammpsPotential(torch.nn.Linear(1, 1))


@pytest.mark.slow
@requires_compile
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_deployed_package_matches_model(representation, tmp_path):
    model = potential(representation).eval()
    path = deploy_lammps(model, str(tmp_path / "model.pt2"), dtype=torch.float64)

    runner = torch._inductor.aoti_load_package(path)
    inputs = flat(periodic(silicon(repeat=2, seed=1)))
    expected = LammpsPotential(model)(**inputs)
    actual = runner(*(inputs[k] for k in LAMMPS_INPUTS))
    for name, a, e in zip(LAMMPS_OUTPUTS, actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10, msg=name)

    metadata = runner.get_metadata()
    assert float(metadata["cutoff"]) == CUTOFF
    assert float(metadata["receptive_field"]) == 2 * CUTOFF
    assert metadata["dtype"] == "float64"
    assert metadata["torch_version"] == torch.__version__
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt2"]


@pytest.mark.slow
@requires_compile
def test_spkdeploy_writes_a_package(tmp_path, monkeypatch):
    torch.save(potential(spk.model.SchNet), tmp_path / "best_model")
    path = tmp_path / "deployed.pt2"
    monkeypatch.setattr(
        sys, "argv", ["spkdeploy", str(tmp_path / "best_model"), str(path)]
    )

    deploy()

    metadata = torch._inductor.aoti_load_package(str(path)).get_metadata()
    assert metadata["dtype"] == "float32"
    assert metadata["device"] == "cpu"


def build_model_check(directory: Path) -> Path:
    """Compile interfaces/lammps/tests/model_check.cpp against this torch."""
    from torch.utils import cpp_extension

    binary = directory / "model_check"
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    subprocess.run(
        ["g++", "-std=c++20", "-O1", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
        + [f"-I{p}" for p in [PAIR_STYLE, *cpp_extension.include_paths()]]
        + [str(PAIR_STYLE / "schnetpack_model.cpp")]
        + [str(PAIR_STYLE / "tests" / "model_check.cpp")]
        + [f"-L{p}" for p in cpp_extension.library_paths()]
        + [f"-Wl,-rpath,{p}" for p in cpp_extension.library_paths()]
        + ["-ltorch", "-ltorch_cpu", "-lc10", "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    return binary


def run_model_check(binary, package, inputs, path, with_offsets=True):
    n_atoms, n_edges = len(inputs["local"]), len(inputs["idx_i"])
    arrays = ["positions", "atomic_numbers", "local", "idx_i", "idx_j"]
    arrays += ["offsets"] if with_offsets else []
    with open(path, "w") as file:
        file.write(f"{n_atoms} {n_edges} {int(with_offsets)}\n")
        for name in arrays:
            values = inputs[name].reshape(-1).tolist()
            file.write(" ".join(repr(v) for v in values) + "\n")
    out = subprocess.run(
        [str(binary), package, str(path)], check=True, capture_output=True, text=True
    ).stdout.split()
    values = torch.tensor([float(v) for v in out], dtype=torch.float64)
    sizes = [1, n_atoms, 3 * n_atoms, 9]
    energy, energy_atom, forces, virial = torch.split(values, sizes)
    return energy[0], energy_atom, forces.reshape(-1, 3), virial.reshape(3, 3)


@pytest.mark.slow
@requires_compile
@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_pair_style_model_runs_the_package_like_python(tmp_path):
    lammps = LammpsPotential(potential(spk.model.PaiNN).eval())
    package = deploy_lammps(
        potential(spk.model.PaiNN).eval(),
        str(tmp_path / "model.pt2"),
        dtype=torch.float64,
    )
    binary = build_model_check(tmp_path)

    cases = [
        (_example(lammps, 1, 0, seed=1), True),
        (_example(lammps, 30, 200, seed=2), True),
        (flat(periodic(silicon(repeat=2))), True),
    ]
    # ghost atoms in domain decomposition have no shifts
    no_shifts = _example(lammps, 20, 120, seed=3)
    no_shifts["offsets"] = torch.zeros_like(no_shifts["offsets"])
    cases.append((no_shifts, False))
    for k, (inputs, with_offsets) in enumerate(cases):
        actual = run_model_check(
            binary, package, inputs, tmp_path / f"in{k}.txt", with_offsets
        )
        expected = lammps(**inputs)
        for name, a, e in zip(LAMMPS_OUTPUTS, actual, expected):
            torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10, msg=name)


@pytest.mark.slow
@requires_compile
@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_pair_style_model_rejects_a_package_of_another_torch(tmp_path):
    package = deploy_lammps(
        potential(spk.model.SchNet).eval(), str(tmp_path / "model.pt2")
    )
    # rewrite the torch version in the package's metadata
    import zipfile

    tampered = tmp_path / "other.pt2"
    with zipfile.ZipFile(package) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith("_metadata.json"):
                data = data.replace(torch.__version__.encode(), b"1.0.0")
            dst.writestr(item, data)
    binary = build_model_check(tmp_path)
    inputs = _example(LammpsPotential(potential(spk.model.SchNet)), 3, 4, seed=0)

    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_model_check(binary, str(tampered), inputs, tmp_path / "in.txt")
    assert "compiled with torch 1.0.0" in failure.value.stderr

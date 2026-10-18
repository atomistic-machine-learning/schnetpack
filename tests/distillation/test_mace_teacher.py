"""The MACE teacher, checked against a scripted stand-in with MACE's interface
and, if mace-torch is installed, against MACE-OFF compiled by the MACE
interface."""

import pickle
import warnings

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from torch.autograd import grad

from schnetpack import properties
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.train.teacher import MaceTeacher, teacher_key

from ..mace.fakes import R_MAX, FakeMace, RecordingMace, reference
from .conftest import molecules_batch, with_neighbors


@pytest.fixture
def fake_path(tmp_path):
    path = str(tmp_path / "fake_mace.pt")
    torch.jit.save(torch.jit.script(FakeMace()), path)
    return path


def rattled(names):
    atoms = [molecule(name) for name in names]
    for seed, structure in enumerate(atoms):
        structure.rattle(0.05, seed=seed)
    return atoms


def periodic():
    """Hydrogen and oxygen in a cell smaller than the cutoff, so that most
    pairs are between periodic images."""
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.3, 0.1], [1.4, 1.6, 1.2]],
        cell=[2.8, 3.1, 2.9],
        pbc=True,
    )
    atoms.rattle(0.05, seed=3)
    return [atoms]


MOLECULES = ["CH4", "H2O", "C2H6"]


def test_matches_the_stand_in_on_molecules(fake_path):
    atoms = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))
    probe = torch.randn_like(batch[properties.R])

    targets = MaceTeacher(model_path=fake_path, cutoff=R_MAX, dtype="float64")(
        batch, probe
    )

    energies, positions = reference(atoms)
    forces = [-grad(e, r, create_graph=True)[0] for e, r in zip(energies, positions)]
    probes = torch.split(probe, [len(a) for a in atoms])
    hvp = [-grad(f, r, grad_outputs=v)[0] for f, r, v in zip(forces, positions, probes)]
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], torch.stack(energies).detach()
    )
    torch.testing.assert_close(
        targets[teacher_key(properties.forces)], torch.cat(forces).detach()
    )
    torch.testing.assert_close(targets[properties.teacher_hvp], torch.cat(hvp))


def test_handles_periodic_images(fake_path):
    atoms = periodic()
    batch = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))

    targets = MaceTeacher(model_path=fake_path, cutoff=R_MAX, dtype="float64")(batch)

    energies, positions = reference(atoms)
    forces = -grad(energies[0], positions[0])[0]
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], torch.stack(energies).detach()
    )
    torch.testing.assert_close(targets[teacher_key(properties.forces)], forces)


def test_runs_in_float32_by_default(fake_path):
    batch = molecules_batch()

    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)
    targets = teacher(batch, torch.randn_like(batch[properties.R]))

    assert teacher.dtype == torch.float32
    assert targets[teacher_key(properties.forces)].dtype == torch.float32
    assert targets[properties.teacher_hvp].dtype == torch.float32


def ethane_lattice(n=25, spacing=6.0):
    """``n`` ethane molecules in one structure, 8 n atoms: its absolute energy
    is tens of keV, far beyond float32's digits for a sub-meV target."""
    structure = Atoms()
    side = int(np.ceil(n ** (1 / 3)))
    for index in range(n):
        ethane = molecule("C2H6")
        ethane.rattle(0.05, seed=index)
        ethane.translate(spacing * np.array(np.unravel_index(index, (side,) * 3)))
        structure += ethane
    return [structure]


@pytest.mark.parametrize("heads", [None, 2])
def test_float32_energies_keep_their_digits(tmp_path, heads):
    """The E0s are summed in float64 and only the interaction energy carries
    float32 error: about 1e-4 eV here, against about 0.1 eV when the E0s are
    summed in float32 too. A per-head table contributes its first head."""
    path = str(tmp_path / "mace.pt")
    torch.jit.save(torch.jit.script(FakeMace(heads=heads)), path)
    atoms = ethane_lattice()
    batch64 = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))
    batch32 = with_neighbors(atoms_to_batch(atoms, dtype=torch.float32))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the MACE model has 2 heads")
        energy = MaceTeacher(model_path=path, cutoff=R_MAX)(batch32)
        exact = MaceTeacher(model_path=path, cutoff=R_MAX, dtype="float64")(batch64)

    key = teacher_key(properties.energy)
    assert batch32[properties.n_atoms].item() == 200
    assert abs(energy[key] - exact[key]).item() < 1e-3


def test_an_archive_without_interaction_energy_is_refused(tmp_path):
    """Its energy would carry the E0s that the teacher adds itself."""
    path = str(tmp_path / "mace.pt")
    torch.jit.save(torch.jit.script(FakeMace(interaction_energy=False)), path)
    batch = with_neighbors(atoms_to_batch(rattled(MOLECULES), dtype=torch.float64))

    with pytest.raises(KeyError, match="interaction_energy"):
        MaceTeacher(model_path=path, cutoff=R_MAX, dtype="float64")(batch)


@pytest.mark.parametrize("with_probe", [True, False])
def test_mace_keeps_the_graph_only_for_a_curvature_target(fake_path, with_probe):
    """MACE's forces keep their graph through its ``training`` argument."""

    batch = molecules_batch()
    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)
    teacher.model.core = RecordingMace(teacher.model.core)

    teacher(batch, torch.randn_like(batch[properties.R]) if with_probe else None)

    assert teacher.model.core.training_argument is with_probe


def test_a_cutoff_below_the_models_r_max_is_refused(fake_path):
    with pytest.raises(ValueError, match="r_max"):
        MaceTeacher(model_path=fake_path, cutoff=R_MAX - 0.5)


def test_elements_the_teacher_does_not_cover_are_refused(fake_path):
    batch = with_neighbors(atoms_to_batch(rattled(["CH4", "NH3"])))

    with pytest.raises(ValueError, match=r"\[7\]"):
        MaceTeacher(model_path=fake_path, cutoff=R_MAX)(batch)


def test_a_missing_archive_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        MaceTeacher(model_path=str(tmp_path / "missing.pt"))


def test_pickles_as_its_configuration(fake_path):
    batch = molecules_batch()
    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)

    restored = pickle.loads(pickle.dumps(teacher))

    torch.testing.assert_close(restored(batch), teacher(batch))


def reference_batches(reference, dtype):
    """The structures of ``mace_off_reference`` as batches, with MACE's energies
    and forces; their match with MACE is the interface's golden test."""
    for name, atoms, energy, forces in reference:
        batch = with_neighbors(atoms_to_batch([atoms], dtype=dtype))
        yield name, batch, energy, forces


def test_a_real_mace_off_archive_gives_curvature(mace_off_archive, mace_off_reference):
    """Curvature against a central difference of the forces, in float64."""
    teacher = MaceTeacher(model_path=mace_off_archive, cutoff=5.0, dtype="float64")
    _, batch, _, _ = next(reference_batches(mace_off_reference, torch.float64))
    probe = torch.randn_like(batch[properties.R])

    targets = teacher(batch, probe)

    eps = 1e-4
    forces_key = teacher_key(properties.forces)
    shifted = [
        teacher({**batch, properties.R: batch[properties.R] + sign * eps * probe})
        for sign in (1, -1)
    ]
    finite_difference = -(shifted[0][forces_key] - shifted[1][forces_key]) / (2 * eps)
    torch.testing.assert_close(
        targets[properties.teacher_hvp], finite_difference, rtol=1e-4, atol=1e-4
    )


def test_a_real_mace_off_archive_runs_in_float32(mace_off_archive, mace_off_reference):
    """The default: a float64 archive cast to float32, double backward included."""
    path = mace_off_archive
    _, batch64, energy, forces = next(
        reference_batches(mace_off_reference, torch.float64)
    )
    batch = with_neighbors(
        {k: v.float() if v.is_floating_point() else v for k, v in batch64.items()}
    )
    probe = torch.randn_like(batch[properties.R])

    targets = MaceTeacher(model_path=path, cutoff=5.0)(batch, probe)
    reference = MaceTeacher(model_path=path, cutoff=5.0, dtype="float64")(
        batch64, probe.double()
    )

    torch.testing.assert_close(
        targets[teacher_key(properties.energy)],
        torch.tensor([float(energy)], dtype=torch.float64),
        rtol=0,
        atol=1e-4,
    )
    torch.testing.assert_close(
        targets[teacher_key(properties.forces)],
        torch.from_numpy(forces).float(),
        rtol=0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        targets[properties.teacher_hvp].double(),
        reference[properties.teacher_hvp],
        rtol=1e-3,
        atol=1e-2,
    )

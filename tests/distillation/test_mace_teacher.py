"""The MACE teacher, checked against an exported stand-in with MACE's
interface and, if mace-torch is installed, against MACE-OFF exported by the
MACE interface. What MaceModel does by itself (periodic images, E0s in float64,
heads, refused inputs) is tested with it; pickling with SchNetPackTeacher."""

import pytest
import torch
from torch.autograd import grad

from schnetpack import properties
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.interfaces.mace_interface import export_mace
from schnetpack.train.teacher import MaceTeacher

from ..mace.fakes import R_MAX, FakeMace, RecordingMace, rattled, reference
from .conftest import absolute_energy, molecules_batch, with_neighbors


@pytest.fixture
def fake_path(tmp_path):
    return export_mace(FakeMace(), str(tmp_path / "fake_mace.pt2"))


MOLECULES = ["CH4", "H2O", "C2H6"]


def test_matches_the_stand_in_on_molecules(fake_path):
    atoms = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))
    probe = torch.randn_like(batch[properties.R])

    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX, dtype="float64")
    targets = teacher(batch, probe)

    energies, positions = reference(atoms)
    forces = [-grad(e, r, create_graph=True)[0] for e, r in zip(energies, positions)]
    probes = torch.split(probe, [len(a) for a in atoms])
    curvature = [
        -grad(f, r, grad_outputs=v)[0] for f, r, v in zip(forces, positions, probes)
    ]
    torch.testing.assert_close(
        absolute_energy(teacher, batch, targets), torch.stack(energies).detach()
    )
    torch.testing.assert_close(
        targets[properties.teacher_forces], torch.cat(forces).detach()
    )
    torch.testing.assert_close(
        targets[properties.teacher_curvature], torch.cat(curvature)
    )


def test_runs_in_float32_by_default(fake_path):
    batch = molecules_batch()

    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)
    targets = teacher(batch, torch.randn_like(batch[properties.R]))

    assert teacher.dtype == torch.float32
    assert targets[properties.teacher_forces].dtype == torch.float32
    assert targets[properties.teacher_curvature].dtype == torch.float32


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
    forces_key = properties.teacher_forces
    shifted = [
        teacher({**batch, properties.R: batch[properties.R] + sign * eps * probe})
        for sign in (1, -1)
    ]
    finite_difference = -(shifted[0][forces_key] - shifted[1][forces_key]) / (2 * eps)
    torch.testing.assert_close(
        targets[properties.teacher_curvature],
        finite_difference,
        rtol=1e-4,
        atol=1e-4,
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

    teacher = MaceTeacher(model_path=path, cutoff=5.0)
    targets = teacher(batch, probe)
    reference = MaceTeacher(model_path=path, cutoff=5.0, dtype="float64")(
        batch64, probe.double()
    )

    torch.testing.assert_close(
        absolute_energy(teacher, batch, targets),
        torch.tensor([float(energy)], dtype=torch.float64),
        rtol=0,
        atol=1e-4,
    )
    torch.testing.assert_close(
        targets[properties.teacher_forces],
        torch.from_numpy(forces).float(),
        rtol=0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        targets[properties.teacher_curvature].double(),
        reference[properties.teacher_curvature],
        rtol=1e-3,
        atol=1e-2,
    )

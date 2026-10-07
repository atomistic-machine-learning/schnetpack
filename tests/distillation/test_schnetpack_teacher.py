"""SchNetPack models as teachers: loading, postprocessing, curvature, pickling."""

import pickle

import pytest
import torch
from ase.build import bulk
from torch import nn

from schnetpack import properties
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.train.teacher import SchNetPackTeacher, teacher_key
from schnetpack.units import convert_units

from .conftest import CUTOFF, make_nnp, molecules_batch, with_neighbors

LOOSE = dict(rtol=1e-4, atol=1e-4)


def fresh(batch):
    """A copy a model may write into without touching ``batch``."""
    return {key: value.clone() for key, value in batch.items()}


def silicon_batch(cutoff=CUTOFF):
    silicon = bulk("Si", "diamond", a=5.43, cubic=True)
    silicon.rattle(0.05, seed=0)
    return with_neighbors(atoms_to_batch([silicon]), cutoff)


def test_loads_a_pickled_model_frozen_in_eval_mode_and_float32(teacher_path):
    teacher = SchNetPackTeacher(teacher_path, model_format="torch")

    assert not teacher.model.training
    assert not any(p.requires_grad for p in teacher.model.parameters())
    assert all(p.dtype == torch.float32 for p in teacher.model.parameters())


def test_predicts_as_the_model_does_postprocessing_included(
    batch, teacher_model, teacher_path
):
    targets = SchNetPackTeacher(teacher_path)(batch)  # auto falls back to torch

    expected = teacher_model.eval()(fresh(batch))
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], expected[properties.energy], **LOOSE
    )
    torch.testing.assert_close(
        targets[teacher_key(properties.forces)],
        expected[properties.forces].float(),
        **LOOSE,
    )


def test_curvature_target_matches_finite_differences_of_the_forces(teacher_path):
    batch = molecules_batch(dtype=torch.float64)
    teacher = SchNetPackTeacher(teacher_path, dtype="float64")
    probe = torch.randn_like(batch[properties.R])
    eps = 1e-4

    def forces_at(displacement):
        displaced = {**batch, properties.R: batch[properties.R] + displacement}
        return teacher(displaced)[teacher_key(properties.forces)]

    finite_difference = (forces_at(-eps * probe) - forces_at(eps * probe)) / (2 * eps)
    torch.testing.assert_close(
        teacher(batch, probe)[properties.teacher_hvp],
        finite_difference,
        rtol=1e-4,
        atol=1e-5,
    )


def test_curvature_pass_leaves_the_teacher_in_eval_mode(batch, teacher_path):
    teacher = SchNetPackTeacher(teacher_path)

    teacher(batch, torch.randn_like(batch[properties.R]))

    assert not teacher.model.training


def test_torchscript_teacher_gives_the_same_targets(tmp_path, batch):
    model = make_nnp(seed=1, energy_mean=-3.0, cast_to_64=False)
    scripted_path = str(tmp_path / "teacher.jit")
    torch.jit.save(torch.jit.script(model), scripted_path)
    pickled_path = str(tmp_path / "teacher.pt")
    torch.save(model, pickled_path)
    probe = torch.randn_like(batch[properties.R])

    scripted = SchNetPackTeacher(scripted_path)  # auto tries TorchScript first
    pickled = SchNetPackTeacher(pickled_path, model_format="torch")

    assert isinstance(scripted.model, torch.jit.ScriptModule)
    torch.testing.assert_close(scripted(batch, probe), pickled(batch, probe), **LOOSE)


def test_rejects_an_unknown_format(teacher_path):
    with pytest.raises(ValueError, match="model_format"):
        SchNetPackTeacher(teacher_path, model_format="onnx")


def test_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="teacher"):
        SchNetPackTeacher(str(tmp_path / "missing.pt"))


def test_warns_about_train_mode_sensitive_modules(tmp_path):
    model = make_nnp(seed=1)
    model.dropout = nn.Dropout(0.1)  # never called; its presence is enough
    path = str(tmp_path / "dropout.pt")
    torch.save(model, path)

    with pytest.warns(UserWarning, match="Dropout"):
        SchNetPackTeacher(path)


def test_pickles_as_its_path(batch, teacher_path):
    teacher = SchNetPackTeacher(teacher_path, cutoff=CUTOFF)

    payload = pickle.dumps(teacher)
    restored = pickle.loads(payload)

    assert len(payload) < len(pickle.dumps(teacher.model)) // 2
    probe = torch.randn_like(batch[properties.R])
    torch.testing.assert_close(restored(batch, probe), teacher(batch, probe))


def test_periodic_list_pruned_to_the_cutoff_gives_the_exact_lists_targets(
    teacher_path,
):
    wide = silicon_batch(cutoff=CUTOFF + 1.0)
    exact = silicon_batch(cutoff=CUTOFF)
    probe = torch.randn_like(exact[properties.R])

    pruned = SchNetPackTeacher(teacher_path, cutoff=CUTOFF)(wide, probe)
    reference = SchNetPackTeacher(teacher_path)(exact, probe)

    torch.testing.assert_close(pruned, reference, **LOOSE)


def test_periodic_batch_in_other_units_gives_converted_targets(teacher_path):
    angstrom = silicon_batch(cutoff=CUTOFF + 1.0)
    to_bohr = convert_units("Ang", "Bohr")
    bohr = {
        **angstrom,
        properties.R: angstrom[properties.R] * to_bohr,
        properties.cell: angstrom[properties.cell] * to_bohr,
        properties.offsets: angstrom[properties.offsets] * to_bohr,
    }
    probe = torch.randn_like(angstrom[properties.R])

    reference = SchNetPackTeacher(teacher_path, cutoff=CUTOFF)(angstrom, probe)
    converted = SchNetPackTeacher(
        teacher_path,
        student_distance_unit="Bohr",
        cutoff=CUTOFF * to_bohr,  # student units: Bohr
    )(bohr, probe)

    angstrom_per_bohr = convert_units("Bohr", "Ang")
    torch.testing.assert_close(
        converted[teacher_key(properties.energy)],
        reference[teacher_key(properties.energy)],
        **LOOSE,
    )
    torch.testing.assert_close(
        converted[teacher_key(properties.forces)],
        reference[teacher_key(properties.forces)] * angstrom_per_bohr,
        **LOOSE,
    )
    torch.testing.assert_close(
        converted[properties.teacher_hvp],
        reference[properties.teacher_hvp] * angstrom_per_bohr**2,
        **LOOSE,
    )

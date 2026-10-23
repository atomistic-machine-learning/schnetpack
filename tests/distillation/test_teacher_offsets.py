"""Energy offsets in distillation: the teacher's energy is relative to its own
offsets per element, and a student trained on it takes the same offsets,
converted to its units, so both are compared in one frame."""

import pytest
import torch
from torch import nn

from schnetpack import properties
from schnetpack.interfaces.mace_interface import export_mace
from schnetpack.model.mace import reference_energies
from schnetpack.objectives import ModelOutput, predict_without_postprocessing
from schnetpack.train.distillation import Distillation
from schnetpack.train.teacher import MaceTeacher, SchNetPackTeacher, TeacherWrapper
from schnetpack.transform import AddOffsets
from schnetpack.units import convert_units

from ..mace.fakes import R_MAX, FakeMace
from .conftest import make_datamodule, make_nnp, molecules_batch

ATOMREFS = torch.zeros(100, dtype=torch.float64)
ATOMREFS[1], ATOMREFS[6], ATOMREFS[7], ATOMREFS[8] = -0.5, -38.0, -54.5, -75.0
#: the teacher's mean energy per atom
MEAN = -3.0


class OpaqueTeacher(TeacherWrapper):
    """A teacher that does not know its offsets."""

    def load(self):
        return None


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """The datamodule writes its split lock to the working directory."""
    monkeypatch.chdir(tmp_path)


def ready(tmp_path, **kwargs):
    """A datamodule set up, as Lightning hands it to the task."""
    datamodule = make_datamodule(tmp_path, **kwargs)
    datamodule.setup("fit")
    return datamodule


def offsets_model(**offsets):
    """A teacher model whose ``AddOffsets`` adds ``offsets``: by default
    ATOMREFS and MEAN."""
    model = make_nnp(seed=1)
    kwargs = dict(
        add_mean=True,
        add_atomrefs=True,
        atomrefs=ATOMREFS.clone(),
        property_mean=torch.tensor([MEAN], dtype=torch.float64),
    )
    model.postprocessors[-1] = AddOffsets(properties.energy, **(kwargs | offsets))
    return model


def saved(model, path):
    torch.save(model, str(path))
    return str(path)


def term(target_property=properties.teacher_energy, name=properties.energy):
    return ModelOutput(
        name=name, target_property=target_property, loss_fn=nn.MSELoss(), metrics={}
    )


# ------------------------------------------------------------- the teachers


def test_a_schnetpack_teacher_adds_its_atomrefs_and_mean(tmp_path):
    path = saved(offsets_model(), tmp_path / "t")

    offsets = SchNetPackTeacher(path).teacher_offsets()

    assert offsets.dtype == torch.float64
    torch.testing.assert_close(offsets, ATOMREFS + MEAN)


def test_only_the_offsets_the_teacher_adds_count(tmp_path):
    path = saved(offsets_model(add_mean=False), tmp_path / "t")

    torch.testing.assert_close(SchNetPackTeacher(path).teacher_offsets(), ATOMREFS)


def test_a_teacher_without_offsets_adds_none_and_warns(tmp_path):
    model = make_nnp(seed=1)
    model.postprocessors = nn.ModuleList()
    teacher = SchNetPackTeacher(saved(model, tmp_path / "t"))

    with pytest.warns(UserWarning, match="zero offsets"):
        offsets = teacher.teacher_offsets()

    assert not offsets.any()


def test_intensive_teacher_offsets_are_refused(tmp_path):
    path = saved(offsets_model(is_extensive=False), tmp_path / "t")

    with pytest.raises(ValueError, match="intensive"):
        SchNetPackTeacher(path).teacher_offsets()


class ScaleShift(nn.Module):
    """MACE's ``ScaleShiftBlock``, as far as the teacher reads it."""

    def __init__(self, shift: float):
        super().__init__()
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float64))


@pytest.mark.parametrize("shift", [None, 0.25])
def test_a_mace_teacher_adds_its_e0s_and_shift(tmp_path, shift):
    core = FakeMace()
    if shift is not None:
        core.scale_shift = ScaleShift(shift)
    path = export_mace(core, str(tmp_path / "mace.pt2"))

    offsets = MaceTeacher(path, cutoff=R_MAX).teacher_offsets()

    expected = reference_energies(FakeMace()) + (shift or 0.0)
    torch.testing.assert_close(offsets, expected)


# --------------------------------------------------- the teacher's energy


def test_the_teacher_energy_is_relative_to_its_offsets(tmp_path):
    """What is left of its energy is what its network predicts before its
    AddOffsets."""
    teacher = SchNetPackTeacher(saved(offsets_model(), tmp_path / "t"))
    batch = molecules_batch()

    energy = teacher(batch)[properties.teacher_energy]

    raw = predict_without_postprocessing(teacher.model, dict(batch))
    assert energy.dtype == torch.float32
    torch.testing.assert_close(energy, raw[properties.energy].float())


def test_a_teacher_without_offsets_returns_absolute_energies(batch):
    class Constant(TeacherWrapper):
        def load(self):
            return None

        def teacher_output(self, inputs, create_graph):
            positions = inputs[properties.R]
            n = len(inputs[properties.n_atoms])
            return {
                properties.energy: torch.full((n,), -100.0, dtype=positions.dtype),
                properties.forces: 0.0 * positions,
            }

    energy = Constant()(batch)[properties.teacher_energy]

    torch.testing.assert_close(energy, torch.full((3,), -100.0))


# ---------------------------------------------------------- the student


@pytest.mark.parametrize("add_mean", [True, False])
def test_the_student_takes_the_teacher_offsets_in_its_units(tmp_path, add_mean):
    """With a mean or without, the student adds what the teacher adds."""
    teacher = SchNetPackTeacher(
        saved(offsets_model(), tmp_path / "t"), teacher_energy_unit="kcal/mol"
    )
    student = make_nnp(atomrefs=True)  # asks to estimate them; no matter
    offsets = student.postprocessors[-1]
    offsets.add_mean = add_mean

    Distillation([term()], student, teacher).setup(ready(tmp_path))

    expected = (ATOMREFS + MEAN) * convert_units("kcal/mol", "eV")
    torch.testing.assert_close(offsets.atomref, expected)
    assert not offsets.mean.any()


def test_a_student_identical_to_its_teacher_predicts_its_absolute_energies(
    tmp_path,
):
    teacher = SchNetPackTeacher(saved(offsets_model(), tmp_path / "t"))
    student = make_nnp(seed=1, atomrefs=True)  # the teacher's weights
    batch = molecules_batch()

    Distillation([term()], student, teacher).setup(None)

    torch.testing.assert_close(
        student(dict(batch))[properties.energy],
        teacher.model(dict(batch))[properties.energy],
    )


def test_the_atomrefs_keep_the_students_size(tmp_path):
    path = export_mace(FakeMace(), str(tmp_path / "fake.pt2"))
    student = make_nnp(atomrefs=True)

    Distillation([term()], student, MaceTeacher(path, cutoff=R_MAX)).setup(None)

    assert len(student.postprocessors[-1].atomref) == 100


def test_teacher_offsets_beyond_the_students_elements_are_refused(tmp_path):
    teacher = SchNetPackTeacher(saved(offsets_model(), tmp_path / "t"))
    student = make_nnp()
    student.postprocessors[-1] = AddOffsets(
        properties.energy, add_atomrefs=True, zmax=10
    )

    with pytest.raises(ValueError, match="zmax=10"):
        Distillation([term()], student, teacher).setup(None)


def test_a_teacher_without_offsets_gives_the_student_none(tmp_path):
    student = make_nnp(atomrefs=True)

    Distillation([term()], student, OpaqueTeacher()).setup(None)

    assert not student.postprocessors[-1].atomref.any()


def test_without_a_teacher_energy_the_datamodule_fits_the_offsets(tmp_path):
    """Nothing names the student's energy; its offsets come from the labels."""
    datamodule = ready(
        tmp_path, load_properties=["energy"], energy=lambda atoms: 2.0 * len(atoms)
    )
    student = make_nnp()  # AddOffsets(add_mean=True)
    teacher = SchNetPackTeacher(saved(offsets_model(), tmp_path / "t"))

    Distillation(
        [term(properties.teacher_forces, properties.forces)], student, teacher
    ).setup(datamodule)

    torch.testing.assert_close(
        student.postprocessors[-1].mean.reshape(1),
        torch.tensor([2.0], dtype=torch.float64),
    )

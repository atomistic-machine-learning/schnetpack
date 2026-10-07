"""Energy statistics for the student's offsets: from the labels when a label
trains the student's energy, otherwise read off the teacher, on a capped
subset of the training split, and stored with the datamodule's statistics."""

import pytest
import torch
from ase.build import molecule
from torch import nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.objectives import ModelOutput, UnsupervisedModelOutput
from schnetpack.train.distillation import (
    TeacherStats,
    student_energy_offsets,
    student_stats_source,
)
from schnetpack.train.teacher import SchNetPackTeacher, TeacherWrapper, teacher_key
from schnetpack.units import convert_units

from .conftest import (
    MOLECULES,
    atoms_to_batch,
    make_datamodule,
    make_nnp,
    molecules_batch,
)

ATOMREFS = torch.zeros(100, dtype=torch.float64)
ATOMREFS[1], ATOMREFS[6], ATOMREFS[8] = -0.5, -38.0, -75.0

#: the molecules of make_datamodule's training split
TRAIN = [MOLECULES[index % len(MOLECULES)] for index in range(8)]


class ElementEnergies(nn.Module):
    """A SchNetPack-style model whose energy is a sum of per-element constants."""

    def __init__(self):
        super().__init__()
        self.register_buffer("atomrefs", ATOMREFS.clone())

    def forward(self, inputs):
        positions = inputs[properties.R]
        per_atom = self.atomrefs[inputs[properties.Z]] + 0.0 * positions.sum(-1)
        n_structures = inputs[properties.n_atoms].shape[0]
        energy = torch.zeros(n_structures, dtype=per_atom.dtype).index_add(
            0, inputs[properties.idx_m], per_atom
        )
        (dEdR,) = grad(energy.sum(), positions, create_graph=True)
        return {properties.energy: energy, properties.forces: -dEdR}


class ElementTeacher(TeacherWrapper):
    """A SchNetPack-style model: its batch in, its energy and forces out. It
    records the structures it was run on."""

    def __init__(self, **kwargs):
        self._seen = []
        super().__init__(**kwargs)

    def load(self):
        return ElementEnergies()

    def teacher_output(self, inputs, create_graph):
        return self.model(inputs)

    def __call__(self, batch, probe=None):
        self._seen += batch[properties.idx].tolist()
        return super().__call__(batch, probe)


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """The datamodule writes its split lock to the working directory."""
    monkeypatch.chdir(tmp_path)


def ready(tmp_path, **kwargs):
    """A datamodule set up, as Lightning hands it to the task."""
    datamodule = make_datamodule(tmp_path, **kwargs)
    datamodule.setup("fit")
    return datamodule


def energies_of(names):
    """The energies of the molecules ``names`` under ATOMREFS, and their sizes."""
    numbers = [torch.tensor(molecule(name).numbers) for name in names]
    energies = torch.stack([ATOMREFS[z].sum() for z in numbers])
    n_atoms = torch.tensor([len(z) for z in numbers], dtype=torch.float64)
    return energies, n_atoms


def term(target_property, cls=ModelOutput):
    return cls(
        name=properties.energy,
        target_property=target_property,
        loss_fn=nn.MSELoss(),
        metrics={},
    )


def test_atomrefs_are_estimated_from_the_teacher(tmp_path):
    stats = TeacherStats(ready(tmp_path), ElementTeacher())

    atomref = stats.get_atomrefs(properties.energy, is_extensive=True)[
        properties.energy
    ]

    torch.testing.assert_close(
        atomref[[1, 6, 7, 8]], ATOMREFS[[1, 6, 7, 8]], rtol=1e-6, atol=1e-5
    )
    assert torch.all(atomref[[0, 2, 9, 99]] == 0)


def test_mean_and_std_per_atom_of_the_teacher_energies(tmp_path):
    stats = TeacherStats(ready(tmp_path), ElementTeacher())

    mean, std = stats.get_stats(
        properties.energy, divide_by_atoms=True, remove_atomref=False
    )

    energies, n_atoms = energies_of(TRAIN)
    torch.testing.assert_close(mean, (energies / n_atoms).mean())
    torch.testing.assert_close(std, (energies / n_atoms).std(correction=0))


def test_removing_the_atomrefs_leaves_nothing(tmp_path):
    stats = TeacherStats(ready(tmp_path), ElementTeacher())

    mean, std = stats.get_stats(
        properties.energy, divide_by_atoms=True, remove_atomref=True
    )

    assert abs(mean.item()) < 1e-5
    assert std.item() < 1e-5


def test_other_properties_are_left_to_the_datamodule(tmp_path):
    """A student whose energy is ``E``: the label ``energy`` is not its energy."""
    datamodule = ready(
        tmp_path, load_properties=["energy"], energy=lambda atoms: -2.0 * len(atoms)
    )
    teacher = ElementTeacher(student_energy_key="E")

    mean, _ = TeacherStats(datamodule, teacher).get_stats(
        properties.energy, True, False
    )

    assert mean.item() == pytest.approx(-2.0)
    assert teacher._seen == []


def test_answers_for_the_students_own_energy_key(tmp_path):
    stats = TeacherStats(ready(tmp_path), ElementTeacher(student_energy_key="E"))

    atomref = stats.get_atomrefs("E", is_extensive=True)["E"]

    torch.testing.assert_close(
        atomref[[1, 6, 8]], ATOMREFS[[1, 6, 8]], rtol=1e-6, atol=1e-5
    )


def test_the_teacher_runs_once_and_only_when_asked(tmp_path):
    teacher = ElementTeacher()
    stats = TeacherStats(ready(tmp_path), teacher)
    assert teacher._seen == []

    stats.get_atomrefs(properties.energy, True)
    stats.get_stats(properties.energy, True, True)

    assert sorted(teacher._seen) == list(range(8))  # the training split, once


def test_the_pass_is_capped_at_max_structures(tmp_path):
    teacher = ElementTeacher()

    TeacherStats(ready(tmp_path), teacher, max_structures=3).get_stats(
        properties.energy, True, False
    )

    assert len(teacher._seen) == 3
    assert set(teacher._seen) <= set(range(8))  # training structures only


def test_the_same_seed_picks_the_same_structures(tmp_path):
    """Every rank and every run fit the same offsets."""
    datamodule = ready(tmp_path, stats_file=None)  # both run the teacher
    teachers = [ElementTeacher(), ElementTeacher()]

    for teacher in teachers:
        TeacherStats(datamodule, teacher, max_structures=3, seed=7).get_stats(
            properties.energy, True, False
        )

    assert sorted(teachers[0]._seen) == sorted(teachers[1]._seen)


def test_another_source_reads_the_stored_statistics(tmp_path):
    """A rerun, a resume or another rank: the teacher does not run again."""
    datamodule = ready(tmp_path)
    first, second = ElementTeacher(), ElementTeacher()

    answers = []
    for teacher in (first, second):
        stats = TeacherStats(datamodule, teacher)
        atomref = stats.get_atomrefs(properties.energy, True)[properties.energy]
        answers.append((atomref, stats.get_stats(properties.energy, True, True)))

    assert first._seen and not second._seen
    torch.testing.assert_close(answers[0], answers[1])


def test_a_teacher_in_other_units_does_not_read_them(tmp_path):
    datamodule = ready(tmp_path)
    TeacherStats(datamodule, ElementTeacher()).get_stats(properties.energy, True, False)
    teacher = ElementTeacher(student_energy_unit="kcal/mol")

    mean, _ = TeacherStats(datamodule, teacher).get_stats(
        properties.energy, True, False
    )

    energies, n_atoms = energies_of(TRAIN)
    expected = (energies / n_atoms).mean() * convert_units("eV", "kcal/mol")
    assert teacher._seen
    torch.testing.assert_close(mean, expected)


def test_a_changed_teacher_file_is_run_again(tmp_path, teacher_path):
    """Same path, other contents: a retrained teacher."""
    datamodule = ready(tmp_path)
    before, _ = TeacherStats(datamodule, SchNetPackTeacher(teacher_path)).get_stats(
        properties.energy, True, False
    )
    torch.save(make_nnp(seed=5, energy_mean=-7.0), teacher_path)

    after, _ = TeacherStats(datamodule, SchNetPackTeacher(teacher_path)).get_stats(
        properties.energy, True, False
    )

    assert after.item() != pytest.approx(before.item())


def test_nothing_is_stored_without_a_stats_file(tmp_path):
    datamodule = ready(tmp_path, stats_file=None)
    first, second = ElementTeacher(), ElementTeacher()

    for teacher in (first, second):
        TeacherStats(datamodule, teacher).get_stats(properties.energy, True, False)

    assert first._seen and second._seen


def test_initializes_the_students_offsets(tmp_path):
    student = make_nnp()  # AddOffsets(add_mean=True), not yet initialized

    student.initialize_transforms(TeacherStats(ready(tmp_path), ElementTeacher()))

    batch = molecules_batch()
    energies, n_atoms = energies_of(TRAIN)
    torch.testing.assert_close(
        student_energy_offsets(student, batch),
        (energies / n_atoms).mean() * batch[properties.n_atoms].double(),
    )


def test_atomref_offsets_reproduce_the_teacher_energies(tmp_path):
    """The teacher's energy is a sum of atomrefs: once they are fitted, nothing
    is left for the mean, whatever the composition."""
    student = make_nnp(atomrefs=True)

    student.initialize_transforms(TeacherStats(ready(tmp_path), ElementTeacher()))

    energies, _ = energies_of(["CH4", "H2O", "C2H6"])  # those of molecules_batch
    torch.testing.assert_close(
        student_energy_offsets(student, molecules_batch()),
        energies,
        rtol=1e-6,
        atol=1e-5,
    )


def test_atomref_offsets_cope_with_one_composition(tmp_path):
    """With a single composition the atomrefs are underdetermined; the offsets
    must still come to the teacher's energy."""
    student = make_nnp(atomrefs=True)

    student.initialize_transforms(
        TeacherStats(ready(tmp_path, molecules=("H2O",)), ElementTeacher())
    )

    batch = atoms_to_batch([molecule("H2O"), molecule("H2O")])
    energies, _ = energies_of(["H2O", "H2O"])
    torch.testing.assert_close(
        student_energy_offsets(student, batch), energies, rtol=1e-6, atol=1e-5
    )


def test_a_label_on_the_students_energy_fits_its_offsets(tmp_path):
    datamodule = make_datamodule(tmp_path)
    outputs = [term(properties.energy), term(teacher_key(properties.energy))]

    assert student_stats_source(outputs, datamodule, ElementTeacher()) is datamodule


def test_without_a_label_the_teacher_fits_the_offsets(tmp_path):
    """Teacher targets and a regularizer on the energy are no labels."""
    datamodule = make_datamodule(tmp_path)
    outputs = [
        term(teacher_key(properties.energy)),
        term(properties.energy, cls=UnsupervisedModelOutput),
    ]

    source = student_stats_source(
        outputs, datamodule, ElementTeacher(), max_structures=5
    )

    assert isinstance(source, TeacherStats)
    assert source.max_structures == 5

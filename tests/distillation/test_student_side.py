"""What the student side of distillation needs: probes, its own curvature, its
energy offsets, and the checks that refuse an unworkable setup."""

import pytest
import torch
from torch import nn

from schnetpack import properties
from schnetpack.objectives import ModelOutput
from schnetpack.train.distillation import (
    check_distillation_setup,
    draw_probe,
    needs_curvature,
    needs_teacher,
    student_energy_offsets,
    student_hvp,
)
from schnetpack.train.teacher import SchNetPackTeacher, teacher_key
from schnetpack.transform import AddOffsets, CastTo64

from .conftest import CUTOFF, make_nnp, molecules_batch, with_neighbors


def output(name, target_property=None):
    return ModelOutput(
        name=name, target_property=target_property, loss_fn=nn.MSELoss(), metrics={}
    )


def test_probe_is_shaped_like_the_positions(batch):
    probe = draw_probe(batch[properties.R])

    assert probe.shape == batch[properties.R].shape
    assert probe.dtype == batch[properties.R].dtype


def test_seeded_probes_repeat_and_are_standard_normal():
    positions = torch.zeros(2000, 3)

    first = draw_probe(positions, torch.Generator().manual_seed(7))
    second = draw_probe(positions, torch.Generator().manual_seed(7))

    assert torch.equal(first, second)
    assert abs(first.mean().item()) < 0.05
    assert abs(first.std().item() - 1.0) < 0.05  # not normalized per structure


def test_student_hvp_matches_finite_differences_of_its_forces():
    batch = molecules_batch(dtype=torch.float64)
    model = make_nnp(seed=0).double().train()
    probe = torch.randn_like(batch[properties.R])
    eps = 1e-4

    def forces_at(displacement):
        displaced = {**batch, properties.R: batch[properties.R] + displacement}
        return model(displaced)[properties.forces].detach()

    inputs = {**batch, properties.R: batch[properties.R].clone()}
    forces = model(inputs)[properties.forces]
    hvp = student_hvp(forces, inputs[properties.R], probe, create_graph=False)

    finite_difference = (forces_at(-eps * probe) - forces_at(eps * probe)) / (2 * eps)
    torch.testing.assert_close(hvp, finite_difference, rtol=1e-4, atol=1e-5)


def test_student_hvp_refuses_forces_without_a_graph(batch):
    model = make_nnp().eval()
    inputs = {key: value.clone() for key, value in batch.items()}
    forces = model(inputs)[properties.forces]

    with pytest.raises(RuntimeError, match="train mode"):
        student_hvp(
            forces,
            inputs[properties.R],
            torch.randn_like(inputs[properties.R]),
            create_graph=False,
        )


def test_energy_offsets_are_what_add_offsets_adds(batch):
    model = make_nnp(energy_mean=-3.0)

    offsets = student_energy_offsets(model, batch)

    assert offsets.dtype == torch.float64
    torch.testing.assert_close(offsets, -3.0 * batch[properties.n_atoms].double())


def test_energy_offsets_include_atomrefs(batch):
    atomref = torch.zeros(100)
    atomref[1], atomref[6], atomref[8] = -0.5, -38.0, -75.0
    model = make_nnp()
    model.postprocessors.append(
        AddOffsets(properties.energy, add_atomrefs=True, atomrefs=atomref)
    )

    offsets = student_energy_offsets(model, batch)

    expected = torch.zeros(3, dtype=torch.float64).index_add(
        0, batch[properties.idx_m], atomref[batch[properties.Z]].double()
    )
    torch.testing.assert_close(offsets, expected)


def test_no_offsets_without_add_offsets(batch):
    model = make_nnp()
    model.postprocessors = nn.ModuleList([CastTo64()])

    assert torch.equal(
        student_energy_offsets(model, batch), torch.zeros(3, dtype=torch.float64)
    )


def test_what_the_outputs_need():
    plain = [output(properties.energy), output(properties.forces)]
    distilled = [
        output(properties.energy, teacher_key(properties.energy)),
        output(properties.hvp, properties.teacher_hvp),
    ]

    assert not needs_teacher(plain) and not needs_curvature(plain)
    assert needs_teacher(distilled) and needs_curvature(distilled)


def test_teacher_targets_without_a_teacher_are_refused():
    with pytest.raises(ValueError, match="no teacher"):
        check_distillation_setup(
            [output(properties.forces, teacher_key(properties.forces))],
            make_nnp(),
            teacher=None,
        )


def test_a_teacher_target_the_teacher_does_not_supply_is_refused(teacher_path):
    """The outputs name ``teacher_<student key>``; a stale default is caught."""
    model = make_nnp(energy_key="E", force_key="F")
    teacher = SchNetPackTeacher(
        teacher_path, student_energy_key="E", student_force_key="F"
    )

    with pytest.raises(ValueError, match="teacher_E"):
        check_distillation_setup(
            [output("E", teacher_key(properties.energy))], model, teacher
        )
    check_distillation_setup([output("E", teacher_key("E"))], model, teacher)


def test_student_keys_the_student_does_not_produce_are_refused(teacher_path):
    teacher = SchNetPackTeacher(teacher_path, student_force_key="F")

    with pytest.raises(ValueError, match="'F'"):
        check_distillation_setup(
            [output(properties.energy, teacher_key(properties.energy))],
            make_nnp(),
            teacher,
        )


def test_a_student_with_dropout_cannot_take_curvature(teacher_path):
    model = make_nnp()
    model.dropout = nn.Dropout(0.1)
    teacher = SchNetPackTeacher(teacher_path)

    with pytest.raises(ValueError, match="Dropout"):
        check_distillation_setup(
            [output(properties.hvp, properties.teacher_hvp)], model, teacher
        )
    # without curvature the student never has to run in train mode
    check_distillation_setup(
        [output(properties.forces, teacher_key(properties.forces))], model, teacher
    )


def test_a_student_pruning_a_longer_list_predicts_as_on_its_own_list():
    """The student prunes the shared list with FilterShortRange; energy, forces
    and curvature must be those of a list at its own cutoff."""
    exact = molecules_batch(dtype=torch.float64)
    wide = with_neighbors(exact, CUTOFF + 1.0)  # replaces the pair entries
    probe = torch.randn_like(exact[properties.R])

    def predict(model, batch):
        inputs = {**batch, properties.R: batch[properties.R].clone()}
        out = model.double().train()(inputs)
        hvp = student_hvp(
            out[properties.forces], inputs[properties.R], probe, create_graph=False
        )
        return out[properties.energy], out[properties.forces], hvp

    pruned = predict(make_nnp(seed=0, filter_cutoff=CUTOFF), wide)
    reference = predict(make_nnp(seed=0), exact)

    torch.testing.assert_close(pruned, reference)

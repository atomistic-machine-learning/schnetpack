"""What the student side of distillation needs: probes, its own curvature, its
keys, and the checks that refuse an unworkable setup."""

import pytest
import torch
from torch import nn

from schnetpack import properties
from schnetpack.objectives import ModelOutput
from schnetpack.train.distillation import (
    Distillation,
    draw_probe,
    needs_curvature,
    needs_teacher,
)
from schnetpack.train.teacher import SchNetPackTeacher, curvature_product
from schnetpack.transform import CastTo64

from .conftest import (
    CUTOFF,
    make_direct_nnp,
    make_nnp,
    molecules_batch,
    with_dropout,
    with_neighbors,
)


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


def test_curvature_product_refuses_forces_without_a_graph(batch):
    model = make_nnp().eval()
    inputs = {key: value.clone() for key, value in batch.items()}
    forces = model(inputs)[properties.forces]

    with pytest.raises(RuntimeError, match="student's forces are not differentiable"):
        curvature_product(
            forces,
            inputs[properties.R],
            torch.randn_like(inputs[properties.R]),
            model="student",
        )


def test_what_the_outputs_need():
    plain = [output(properties.energy), output(properties.forces)]
    distilled = [
        output(properties.energy, properties.teacher_energy),
        output(properties.curvature, properties.teacher_curvature),
    ]

    assert not needs_teacher(plain) and not needs_curvature(plain)
    assert needs_teacher(distilled) and needs_curvature(distilled)


def test_two_student_outputs_on_one_teacher_target_are_refused(teacher_path):
    outputs = [
        output(properties.forces, properties.teacher_forces),
        output("other_forces", properties.teacher_forces),
    ]

    with pytest.raises(ValueError, match="one student output per teacher target"):
        Distillation(outputs, make_nnp(), SchNetPackTeacher(teacher_path))


def test_curvature_without_a_force_term_is_refused(teacher_path):
    with pytest.raises(ValueError, match="teacher_forces"):
        Distillation(
            [output(properties.curvature, properties.teacher_curvature)],
            make_nnp(),
            SchNetPackTeacher(teacher_path),
        )


def test_student_keys_the_student_does_not_produce_are_refused(teacher_path):
    with pytest.raises(ValueError, match="'F'"):
        Distillation(
            [output("F", properties.teacher_forces)],
            make_nnp(),
            SchNetPackTeacher(teacher_path),
        )


def test_labels_and_teacher_on_one_energy_are_refused(teacher_path):
    outputs = [
        output(properties.energy),
        output(properties.energy, properties.teacher_energy),
    ]

    with pytest.raises(ValueError, match="at once"):
        Distillation(outputs, make_nnp(atomrefs=True), SchNetPackTeacher(teacher_path))


def test_a_teacher_energy_needs_add_offsets_with_atomrefs(teacher_path):
    """The teacher's energy is relative to its offsets per element; a mean
    cannot hold them."""
    teacher = SchNetPackTeacher(teacher_path)
    outputs = [output(properties.energy, properties.teacher_energy)]
    without = make_nnp()
    without.postprocessors = nn.ModuleList([CastTo64()])

    with pytest.raises(ValueError, match="add_atomrefs=True"):
        Distillation(outputs, make_nnp(atomrefs=False), teacher)
    with pytest.raises(ValueError, match="add_atomrefs=True"):
        Distillation(outputs, without, teacher)
    Distillation(outputs, make_nnp(atomrefs=True), teacher)


def test_a_student_with_dropout_cannot_take_curvature(teacher_path):
    model = make_nnp()
    model.dropout = nn.Dropout(0.1)
    teacher = SchNetPackTeacher(teacher_path)

    with pytest.raises(ValueError, match="Dropout"):
        Distillation(
            [
                output(properties.forces, properties.teacher_forces),
                output(properties.curvature, properties.teacher_curvature),
            ],
            model,
            teacher,
        )
    # without curvature the student never has to run in train mode
    Distillation([output(properties.forces, properties.teacher_forces)], model, teacher)


def test_curvature_product_is_the_hessian_product_for_conservative_forces():
    """F = -dE/dR with E = R^T A R / 2 for symmetric A: the product is A v."""
    a = torch.randn(6, 6, dtype=torch.float64)
    a = a + a.T
    positions = torch.randn(6, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(6, dtype=torch.float64)
    forces = -(a @ positions)

    torch.testing.assert_close(curvature_product(forces, positions, probe), a @ probe)


def test_curvature_product_is_the_transposed_jacobian_product_for_direct_forces():
    """For F = B R with any B, the product is -B^T v."""
    b = torch.randn(6, 6, dtype=torch.float64)
    positions = torch.randn(6, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(6, dtype=torch.float64)

    torch.testing.assert_close(
        curvature_product(b @ positions, positions, probe), -(b.T @ probe)
    )


def test_without_a_teacher_plain_outputs_need_no_distillation():
    plain = [output(properties.energy), output(properties.forces)]

    assert Distillation.optional(plain, make_nnp(), teacher=None) is None


def test_needs_grad_covers_the_students_derivatives_and_curvature(teacher_path):
    teacher = SchNetPackTeacher(teacher_path)
    forces = [output(properties.forces, properties.teacher_forces)]
    curvature = forces + [output(properties.curvature, properties.teacher_curvature)]

    assert Distillation(forces, make_nnp(), teacher).needs_grad
    assert Distillation(curvature, make_direct_nnp(), teacher).needs_grad
    assert not Distillation(forces, make_direct_nnp(), teacher).needs_grad


def test_a_direct_force_student_with_dropout_takes_curvature_in_eval_mode(
    batch, teacher_path
):
    """Its forces keep their graph in eval mode, so it is not switched to train
    mode, and its dropout stays off."""
    student = with_dropout(make_direct_nnp()).eval()
    outputs = [
        output(properties.forces, properties.teacher_forces),
        output(properties.curvature, properties.teacher_curvature),
    ]
    distillation = Distillation(outputs, student, SchNetPackTeacher(teacher_path))

    first, _ = distillation.step(dict(batch), train=False)
    second, _ = distillation.step(dict(batch), train=False)

    assert not student.training
    torch.testing.assert_close(first[properties.energy], second[properties.energy])
    assert first[properties.curvature].shape == batch[properties.R].shape


def test_a_student_pruning_a_longer_list_predicts_as_on_its_own_list():
    """The student prunes the shared list with FilterShortRange; energy, forces
    and curvature must be those of a list at its own cutoff."""
    exact = molecules_batch(dtype=torch.float64)
    wide = with_neighbors(exact, CUTOFF + 1.0)  # replaces the pair entries
    probe = torch.randn_like(exact[properties.R])

    def predict(model, batch):
        inputs = {**batch, properties.R: batch[properties.R].clone()}
        out = model.double().train()(inputs)
        curvature = curvature_product(
            out[properties.forces], inputs[properties.R], probe
        )
        return out[properties.energy], out[properties.forces], curvature

    pruned = predict(make_nnp(seed=0, filter_cutoff=CUTOFF), wide)
    reference = predict(make_nnp(seed=0), exact)

    torch.testing.assert_close(pruned, reference)

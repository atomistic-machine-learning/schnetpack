"""One distillation step, end to end: targets from batch and teacher, the
student's curvature, and the loss a hand-written loop trains on."""

import copy

import pytest
import torch
from torch import nn

from schnetpack import properties
from schnetpack.objectives import AtomMask, ModelOutput, calculate_loss
from schnetpack.train.distillation import Distillation, draw_probe
from schnetpack.train.teacher import SchNetPackTeacher

from .conftest import force_jacobian, make_direct_nnp, make_nnp, molecules_batch


def distillation_loss(outputs, student, teacher, batch, train=True):
    """The loss of a hand-written distillation loop; outside training, on the
    probes of seed 0."""
    distillation = Distillation(outputs, student, teacher)
    pred, targets = distillation.step(batch, train=train)
    return calculate_loss(outputs, pred, targets)


class Explode(nn.Module):
    """An input module that fails, to interrupt the student mid-forward."""

    def forward(self, inputs):
        raise RuntimeError("student failed")


def distillation_outputs(energy_weight=1.0, forces_weight=1.0, curvature_weight=1.0):
    def term(name, target, weight):
        return ModelOutput(
            name=name,
            target_property=target,
            loss_fn=nn.MSELoss(),
            loss_weight=weight,
            metrics={},
        )

    return [
        term(properties.energy, properties.teacher_energy, energy_weight),
        term(properties.forces, properties.teacher_forces, forces_weight),
        term(properties.curvature, properties.teacher_curvature, curvature_weight),
    ]


def test_a_student_identical_to_its_teacher_has_zero_loss(batch, teacher_path):
    """Under keys of its own, too: curvature finds the student's forces by the
    term trained on the teacher's."""
    student = make_nnp(
        seed=1, energy_mean=-3.0, energy_key="E", force_key="F", atomrefs=True
    )
    teacher = SchNetPackTeacher(teacher_path)
    outputs = [
        ModelOutput(name=name, target_property=target, loss_fn=nn.MSELoss(), metrics={})
        for name, target in (
            ("E", properties.teacher_energy),
            ("F", properties.teacher_forces),
            (properties.curvature, properties.teacher_curvature),
        )
    ]

    loss = distillation_loss(outputs, student, teacher, batch)

    assert loss.item() < 1e-8


def test_the_curvature_term_trains_the_student_and_only_the_student(
    batch, teacher_path
):
    student = make_nnp(seed=2)
    teacher = SchNetPackTeacher(teacher_path)

    distillation_loss(curvature_outputs(), student, teacher, batch).backward()

    assert all(p.grad is not None for p in student.representation.parameters())
    assert any(p.grad.abs().sum() > 0 for p in student.representation.parameters())
    assert all(p.grad is None for p in teacher.model.parameters())


def curvature_outputs():
    """A curvature term, and a force term of weight 0 naming the forces it is
    taken through."""
    return [
        ModelOutput(
            name=properties.curvature,
            target_property=properties.teacher_curvature,
            loss_fn=nn.MSELoss(),
            metrics={},
        ),
        ModelOutput(
            name=properties.forces,
            target_property=properties.teacher_forces,
            loss_fn=nn.MSELoss(),
            loss_weight=0.0,
            metrics={},
        ),
    ]


def test_a_student_predicting_forces_directly_takes_curvature(batch, teacher_path):
    """Without a Forces module nothing asks for position gradients; its
    curvature is the vector-Jacobian product of its forces, ``-(dF/dR)^T v``."""
    student = make_direct_nnp(seed=2).double()
    batch = {
        key: value.double() if value.is_floating_point() else value
        for key, value in batch.items()
    }
    jacobian = force_jacobian(student, batch)
    probe = draw_probe(batch[properties.R], torch.Generator().manual_seed(0))

    pred, _ = Distillation(
        curvature_outputs(), student, SchNetPackTeacher(teacher_path), probe_seed=0
    ).step(batch, train=False)

    expected = -(jacobian.T @ probe.flatten()).reshape(probe.shape)
    torch.testing.assert_close(pred[properties.curvature], expected)


def test_the_curvature_term_trains_a_direct_force_student(batch, teacher_path):
    student = make_direct_nnp(seed=2)

    distillation_loss(
        curvature_outputs(), student, SchNetPackTeacher(teacher_path), batch
    ).backward()

    assert any(p.grad.abs().sum() > 0 for p in student.representation.parameters())


def test_a_teacher_predicting_forces_directly_supplies_curvature(batch, tmp_path):
    """The identical student has zero loss, curvature included."""
    teacher_model = make_direct_nnp(seed=1)
    path = tmp_path / "direct_teacher.pt"
    torch.save(teacher_model, path)
    student = copy.deepcopy(teacher_model)

    loss = distillation_loss(
        distillation_outputs()[1:], student, SchNetPackTeacher(str(path)), batch
    )

    assert loss.item() < 1e-8


def test_labels_are_read_before_the_student_overwrites_the_batch(batch, teacher_path):
    label = torch.full((3,), 42.0)
    batch = {**batch, properties.energy: label.clone()}
    outputs = [
        ModelOutput(name=properties.energy, loss_fn=nn.MSELoss(), metrics={}),
        ModelOutput(
            name=properties.forces,
            target_property=properties.teacher_forces,
            loss_fn=nn.MSELoss(),
            metrics={},
        ),
    ]

    pred, targets = Distillation(
        outputs, make_nnp(seed=2), SchNetPackTeacher(teacher_path)
    ).step(batch)

    assert torch.equal(targets[properties.energy], label)
    assert not torch.equal(pred[properties.energy], label)


def test_curvature_in_eval_mode_leaves_the_student_as_it_was(batch, teacher_path):
    student = make_nnp(seed=2, atomrefs=True).eval()

    pred, _ = Distillation(
        distillation_outputs(), student, SchNetPackTeacher(teacher_path)
    ).step(batch, train=False)

    assert pred[properties.curvature].shape == batch[properties.R].shape
    assert not student.training
    assert student.do_postprocessing


def test_a_failing_student_forward_restores_its_mode_and_postprocessing(
    batch, teacher_path
):
    student = make_nnp(seed=2, atomrefs=True).eval()
    student.input_modules.append(Explode())

    with pytest.raises(RuntimeError, match="student failed"):
        Distillation(
            distillation_outputs(), student, SchNetPackTeacher(teacher_path)
        ).step(batch)

    assert not student.training
    assert student.do_postprocessing


def test_a_hand_written_loop_reduces_the_loss(teacher_path):
    torch.manual_seed(0)
    student = make_nnp(seed=2, atomrefs=True)
    teacher = SchNetPackTeacher(teacher_path)
    outputs = distillation_outputs(
        energy_weight=0.01, forces_weight=1.0, curvature_weight=0.01
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    def evaluate():
        return distillation_loss(
            outputs, student, teacher, molecules_batch(), train=False
        ).item()

    initial = evaluate()
    for _ in range(30):
        loss = distillation_loss(outputs, student, teacher, molecules_batch())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert evaluate() < initial


def test_a_mask_restricts_a_teacher_target(batch, teacher_path):
    student = make_nnp(seed=2)
    teacher = SchNetPackTeacher(teacher_path)
    n_atoms = batch[properties.R].shape[0]
    keep = torch.arange(n_atoms) < n_atoms // 2
    batch = {**batch, "considered_atoms": keep}
    forces = ModelOutput(
        name=properties.forces,
        target_property=properties.teacher_forces,
        loss_fn=nn.MSELoss(),
        metrics={},
        masks=[AtomMask("considered_atoms")],
    )

    loss = distillation_loss([forces], student, teacher, dict(batch))
    pred, targets = Distillation([forces], student, teacher).step(dict(batch))

    expected = nn.functional.mse_loss(
        pred[properties.forces][keep], targets[properties.teacher_forces][keep]
    )
    assert torch.isclose(loss, expected)

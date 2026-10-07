"""One distillation step, end to end: targets from batch and teacher, the
student's curvature, and the loss a hand-written loop trains on."""

import copy

import pytest
import torch
from torch import nn

from schnetpack import properties
from schnetpack.objectives import AtomMask, ModelOutput, calculate_loss
from schnetpack.train.distillation import (
    check_distillation_setup,
    distillation_predictions,
)
from schnetpack.train.teacher import SchNetPackTeacher, teacher_key

from .conftest import make_nnp, molecules_batch


def distillation_loss(outputs, student, teacher, batch, generator=None):
    """The loss of a hand-written distillation loop (ADR-0025)."""
    check_distillation_setup(outputs, student, teacher)
    pred, targets = distillation_predictions(
        outputs, student, teacher, batch, generator=generator
    )
    return calculate_loss(outputs, pred, targets)


class Explode(nn.Module):
    """An input module that fails, to interrupt the student mid-forward."""

    def forward(self, inputs):
        raise RuntimeError("student failed")


def distillation_outputs(energy_weight=1.0, forces_weight=1.0, hvp_weight=1.0):
    def term(name, target, weight):
        return ModelOutput(
            name=name,
            target_property=target,
            loss_fn=nn.MSELoss(),
            loss_weight=weight,
            metrics={},
        )

    return [
        term(properties.energy, teacher_key(properties.energy), energy_weight),
        term(properties.forces, teacher_key(properties.forces), forces_weight),
        term(properties.hvp, properties.teacher_hvp, hvp_weight),
    ]


def test_a_student_identical_to_its_teacher_has_zero_loss(
    batch, teacher_model, teacher_path
):
    student = copy.deepcopy(teacher_model)  # same weights, same offsets

    loss = distillation_loss(
        distillation_outputs(), student, SchNetPackTeacher(teacher_path), batch
    )

    assert loss.item() < 1e-8


def test_a_student_with_its_own_keys_distills(batch, teacher_path):
    """Offsets and curvature find the student's energy and forces by the keys
    the wrapper declares; the identical student has zero loss."""
    student = make_nnp(seed=1, energy_mean=-3.0, energy_key="E", force_key="F")
    teacher = SchNetPackTeacher(
        teacher_path, student_energy_key="E", student_force_key="F"
    )
    outputs = [
        ModelOutput(
            name=name,
            target_property=teacher_key(name),
            loss_fn=nn.MSELoss(),
            metrics={},
        )
        for name in ("E", "F", properties.hvp)
    ]

    loss = distillation_loss(outputs, student, teacher, batch)

    assert loss.item() < 1e-8


def test_the_curvature_term_trains_the_student_and_only_the_student(
    batch, teacher_path
):
    student = make_nnp(seed=2)
    teacher = SchNetPackTeacher(teacher_path)
    outputs = [
        ModelOutput(
            name=properties.hvp,
            target_property=properties.teacher_hvp,
            loss_fn=nn.MSELoss(),
            metrics={},
        )
    ]

    distillation_loss(outputs, student, teacher, batch).backward()

    assert all(p.grad is not None for p in student.representation.parameters())
    assert any(p.grad.abs().sum() > 0 for p in student.representation.parameters())
    assert all(p.grad is None for p in teacher.model.parameters())


def test_labels_are_read_before_the_student_overwrites_the_batch(batch, teacher_path):
    label = torch.full((3,), 42.0)
    batch = {**batch, properties.energy: label.clone()}
    outputs = [
        ModelOutput(name=properties.energy, loss_fn=nn.MSELoss(), metrics={}),
        ModelOutput(
            name=properties.energy,
            target_property=teacher_key(properties.energy),
            loss_fn=nn.MSELoss(),
            metrics={},
        ),
    ]

    pred, targets = distillation_predictions(
        outputs, make_nnp(seed=2), SchNetPackTeacher(teacher_path), batch
    )

    assert torch.equal(targets[properties.energy], label)
    assert not torch.equal(targets[teacher_key(properties.energy)], label)
    assert not torch.equal(pred[properties.energy], label)


def test_curvature_in_eval_mode_leaves_the_student_as_it_was(batch, teacher_path):
    student = make_nnp(seed=2).eval()

    pred, _ = distillation_predictions(
        distillation_outputs(),
        student,
        SchNetPackTeacher(teacher_path),
        batch,
        create_graph=False,
    )

    assert pred[properties.hvp].shape == batch[properties.R].shape
    assert not student.training
    assert student.do_postprocessing


def test_a_failing_student_forward_restores_its_mode_and_postprocessing(
    batch, teacher_path
):
    student = make_nnp(seed=2).eval()
    student.input_modules.append(Explode())

    with pytest.raises(RuntimeError, match="student failed"):
        distillation_predictions(
            distillation_outputs(), student, SchNetPackTeacher(teacher_path), batch
        )

    assert not student.training
    assert student.do_postprocessing


def test_a_seeded_generator_makes_the_loss_repeatable(teacher_path):
    student = make_nnp(seed=2)
    teacher = SchNetPackTeacher(teacher_path)

    losses = [
        distillation_loss(
            distillation_outputs(),
            student,
            teacher,
            molecules_batch(),
            generator=torch.Generator().manual_seed(0),
        ).item()
        for _ in range(2)
    ]

    assert losses[0] == losses[1]


def test_a_hand_written_loop_reduces_the_loss(teacher_path):
    torch.manual_seed(0)
    student = make_nnp(seed=2)
    teacher = SchNetPackTeacher(teacher_path)
    outputs = distillation_outputs(
        energy_weight=0.01, forces_weight=1.0, hvp_weight=0.01
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    def evaluate():
        return distillation_loss(
            outputs,
            student,
            teacher,
            molecules_batch(),
            generator=torch.Generator().manual_seed(0),
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
        target_property=teacher_key(properties.forces),
        loss_fn=nn.MSELoss(),
        metrics={},
        masks=[AtomMask("considered_atoms")],
    )

    loss = distillation_loss([forces], student, teacher, dict(batch))
    pred, targets = distillation_predictions([forces], student, teacher, dict(batch))

    expected = nn.functional.mse_loss(
        pred[properties.forces][keep], targets[teacher_key(properties.forces)][keep]
    )
    assert torch.isclose(loss, expected)

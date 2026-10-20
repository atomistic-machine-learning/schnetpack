"""A teacher in the Lightning task: steps, seeded validation, offsets from the
teacher at setup, checkpoints."""

import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torchmetrics.regression import MeanAbsoluteError

from schnetpack import properties
from schnetpack.lightning import AtomisticTask
from schnetpack.lightning.utils import load_task_from_checkpoint
from schnetpack.objectives import ModelOutput, calculate_loss
from schnetpack.train import Distillation, SchNetPackTeacher

from .conftest import CUTOFF, make_datamodule, make_nnp, molecules_batch
from .test_distillation_loss import distillation_outputs


def make_task(teacher_path, **kwargs):
    return AtomisticTask(
        model=make_nnp(seed=2, filter_cutoff=CUTOFF, atomrefs=True),
        outputs=distillation_outputs(),
        optimizer_args={"lr": 1e-3},
        teacher=SchNetPackTeacher(teacher_path, cutoff=CUTOFF),
        **kwargs,
    )


def term(name, target_property):
    return ModelOutput(
        name=name,
        target_property=target_property,
        loss_fn=torch.nn.MSELoss(),
        metrics={"mae": MeanAbsoluteError()},
    )


def fast_trainer(tmp_path):
    return pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )


def test_the_teacher_is_not_part_of_the_task(teacher_path):
    task = make_task(teacher_path)

    assert len(list(task.parameters())) == len(list(task.model.parameters()))
    assert all(key.startswith(("model.", "outputs.")) for key in task.state_dict())


def test_teacher_targets_without_a_teacher_are_refused():
    with pytest.raises(ValueError, match="no teacher"):
        AtomisticTask(
            model=make_nnp(), outputs=distillation_outputs(), optimizer_args={"lr": 1}
        )


def test_a_training_step_is_the_pure_distillation_loss(teacher_path):
    task = make_task(teacher_path)

    torch.manual_seed(0)
    via_task = task.training_step(molecules_batch(), 0)
    torch.manual_seed(0)
    pred, targets = Distillation(task.outputs, task.model, task.teacher).step(
        molecules_batch()
    )
    plain = calculate_loss(task.outputs, pred, targets)

    torch.testing.assert_close(via_task, plain)


def test_validation_loss_repeats_across_epochs(teacher_path):
    task = make_task(teacher_path)
    task.eval()

    losses = []
    for _ in range(2):
        task.on_validation_epoch_start()
        losses.append(task.validation_step(molecules_batch(), 0)["val_loss"].item())

    assert losses[0] == losses[1]
    assert not task.model.training


def test_trainer_distills_and_sets_the_student_offsets_from_the_teacher(
    tmp_path, monkeypatch, teacher_path
):
    monkeypatch.chdir(tmp_path)
    task = make_task(teacher_path)
    trainer = fast_trainer(tmp_path)

    trainer.fit(task, datamodule=make_datamodule(tmp_path))

    # the labels are zero and not even loaded: the offsets came from the teacher
    offsets = task.model.postprocessors[-1]
    torch.testing.assert_close(offsets.atomref, task.teacher.energy_offsets()[:100])
    assert not offsets.mean.any()
    assert torch.isfinite(trainer.callback_metrics["val_loss"])


def test_checkpoint_restores_the_teacher_from_its_path(
    tmp_path, monkeypatch, teacher_path, batch
):
    monkeypatch.chdir(tmp_path)
    task = make_task(teacher_path)
    trainer = fast_trainer(tmp_path)
    trainer.fit(task, datamodule=make_datamodule(tmp_path))
    checkpoint = str(tmp_path / "task.ckpt")
    trainer.save_checkpoint(checkpoint)

    restored = load_task_from_checkpoint(AtomisticTask, checkpoint)

    probe = torch.randn_like(batch[properties.R])
    torch.testing.assert_close(
        restored.teacher(batch, probe), task.teacher(batch, probe)
    )
    os.rename(teacher_path, teacher_path + ".moved")
    with pytest.raises(FileNotFoundError, match="teacher"):
        load_task_from_checkpoint(AtomisticTask, checkpoint)


def test_a_label_and_a_teacher_target_for_one_output_are_logged_apart(
    tmp_path, monkeypatch, teacher_path
):
    """Force labels next to the teacher's forces: both train the student's
    forces, so their metrics must not share a name."""
    monkeypatch.chdir(tmp_path)

    task = AtomisticTask(
        model=make_nnp(seed=2, energy_mean=-3.0),
        outputs=[
            term(properties.forces, properties.forces),
            term(properties.forces, properties.teacher_forces),
        ],
        optimizer_args={"lr": 1e-3},
        teacher=SchNetPackTeacher(teacher_path),
    )
    trainer = fast_trainer(tmp_path)
    datamodule = make_datamodule(
        tmp_path, ["forces"], forces=lambda atoms: np.zeros((len(atoms), 3))
    )

    trainer.fit(task, datamodule=datamodule)

    assert "val_forces_mae" in trainer.callback_metrics
    assert "val_teacher_forces_mae" in trainer.callback_metrics

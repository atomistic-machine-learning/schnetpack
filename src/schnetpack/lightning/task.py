from typing import Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel
from schnetpack.objectives import (
    ModelOutput,
    calculate_loss,
    extract_targets,
    predict_without_postprocessing,
)
from schnetpack.train.distillation import (
    TEACHER_STATS_SIZE,
    check_distillation_setup,
    distillation_predictions,
    needs_curvature,
    student_stats_source,
)
from schnetpack.train.teacher import TeacherWrapper

__all__ = ["AtomisticTask"]


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    The loss itself is assembled by the pure-PyTorch functions in
    :mod:`schnetpack.objectives`; this class adds what the Lightning Trainer
    needs (logging, optimizer/scheduler configuration, warmup). To train
    without Lightning, use the model and those functions directly in your own
    training loop.
    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: list[ModelOutput],
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: dict[str, Any] | None = None,
        scheduler_cls: type | None = None,
        scheduler_args: dict[str, Any] | None = None,
        scheduler_monitor: str | None = None,
        warmup_steps: int = 0,
        teacher: TeacherWrapper | None = None,
        probe_seed: int = 0,
        teacher_stats_size: int | None = TEACHER_STATS_SIZE,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
            teacher: the teacher of knowledge distillation. Outputs may then be
              trained on its targets, ``teacher.target_keys``: ``teacher_<key>``
              for the student's energy and force keys, and ``teacher_hvp``; see
              :mod:`schnetpack.train.distillation`.
            probe_seed: seed of the probes drawn in validation and testing, so
              their losses depend on it. The probe generator is reset to this
              seed at the start of every validation and test epoch, so each
              epoch sees the same probes and validation losses are comparable
              across epochs. Training draws fresh probes every step.
            teacher_stats_size: training structures the teacher runs on to fit
              the student's energy offsets when no label trains its energy;
              None for the whole split. The result is stored with the
              datamodule's statistics, so reruns and resumes read it
              (ADR-0017).
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args or {}
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.teacher = teacher
        self.probe_seed = probe_seed
        self.teacher_stats_size = teacher_stats_size
        self._probe_generator = torch.Generator().manual_seed(probe_seed)
        check_distillation_setup(self.outputs, self.model, teacher)

        # curvature is a derivative, in validation and testing too
        self.grad_enabled = bool(self.model.required_derivatives) or needs_curvature(
            self.outputs
        )
        self.lr = self.optimizer_kwargs["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            stats = self.trainer.datamodule
            if self.teacher is not None and stats is not None:
                # labels first; without them, the teacher's energies (ADR-0017)
                stats = student_stats_source(
                    self.outputs,
                    stats,
                    self.teacher,
                    device=self.trainer.strategy.root_device,
                    max_structures=self.teacher_stats_size,
                )
            self.model.initialize_transforms(stats)

    def forward(self, inputs: dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.target_property}_{metric_name}",
                    cast(Metric, metric),
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def _step(self, batch, subset):
        """Composite loss of one batch, with the metrics of ``subset`` logged.

        Targets are the labels from the batch; with a teacher they are its
        targets, and the student's curvature is predicted too."""
        if self.teacher is None:
            targets = extract_targets(self.outputs, batch)
            pred = predict_without_postprocessing(self.model, batch)
        else:
            pred, targets = distillation_predictions(
                self.outputs,
                self.model,
                self.teacher,
                batch,
                generator=None if subset == "train" else self._probe_generator,
                create_graph=subset == "train",
            )
        loss = calculate_loss(self.outputs, pred, targets)
        self.log_metrics(pred, targets, subset)
        return loss

    def on_validation_epoch_start(self):
        self._probe_generator.manual_seed(self.probe_seed)

    def on_test_epoch_start(self):
        self._probe_generator.manual_seed(self.probe_seed)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        loss = self._step(batch, "val")
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        loss = self._step(batch, "test")
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        return {"test_loss": loss}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if not self.scheduler_cls:
            return optimizer

        scheduler = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
        scheduler_config: LRSchedulerConfigType = {
            "scheduler": scheduler,
            "name": "lr_schedule",
        }
        if self.schedule_monitor:
            scheduler_config["monitor"] = self.schedule_monitor
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def optimizer_step(
        self,
        epoch: int | None = None,
        batch_idx: int | None = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: bool | None = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            try:
                torch.save(self.model, path)
            finally:
                self.model.do_postprocessing = pp_status

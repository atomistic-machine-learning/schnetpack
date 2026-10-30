import gc
import warnings
from typing import Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from torch import nn as nn
from torchmetrics import Metric

from schnetpack import properties
from schnetpack.model.base import AtomisticModel
from schnetpack.model.compile import CompiledPotential
from schnetpack.objectives import (
    ModelOutput,
    calculate_loss,
    calculate_padded_loss,
    extract_targets,
    padding_mask,
    predict_without_postprocessing,
)
from schnetpack.train.distillation import Distillation
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

    Two flags make training faster (see the user guide, "Training faster"):

    - ``compile``: the model runs as one compiled graph including its derivatives.
    - ``cuda_graph``: the whole training step (building the batch, the transforms,
      the model, the loss and its gradients) is recorded once as a CUDA graph and
      replayed every step: one launch instead of hundreds of kernels, and no host
      synchronization. Needs the data on the GPU
      (:class:`~schnetpack.lightning.AtomsDataModule` with ``data_device``).

    Padded batches (data on the GPU) are handled in every mode: their padding rows
    are left out of losses and metrics.
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
        compile: bool = False,
        cuda_graph: bool = False,
        grad_clip: float | None = None,
        skip_loss_above: float | None = None,
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
              trained on its targets.
            probe_seed: The probe generator is reset to this seed at the start
              of every validation and test epoch, so each epoch sees the same
              probes and validation losses are comparable across epochs. Training
              draws fresh probes every step.
            compile: run the model as a compiled graph including its derivatives (see
              `schnetpack.model.CompiledPotential`, which needs torch >= 2.14). With a
              teacher, whose curvature needs derivatives of the forces, the model
              runs without compilation.
            cuda_graph: run the whole training step as one CUDA graph; needs the
              data on the GPU and compiles the model (with fixed shapes). Takes no
              teacher, and outputs with ``MSELoss`` or ``L1Loss`` and no loss masks,
              or ``VariableOutput``. A fused optimizer
              (``optimizer_args={"fused": True}``) keeps the step free of host syncs.
            grad_clip: clip the norm of the gradients to this value
            skip_loss_above: skip the optimizer step (and the EMA update) of a batch
              whose loss exceeds this value or is not finite
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
        # None without a teacher
        self.distillation = Distillation.optional(
            self.outputs, self.model, teacher, probe_seed
        )

        # curvature is a derivative, in validation and testing too
        self.grad_enabled = (
            self.distillation.needs_grad
            if self.distillation is not None
            else bool(self.model.required_derivatives)
        )
        self.lr = self.optimizer_kwargs["lr"]
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.skip_loss_above = skip_loss_above
        self.cuda_graph = cuda_graph

        self.compiled_model = None
        if cuda_graph:
            if self.distillation is not None:
                raise ValueError("cuda_graph does not take a teacher")
            # fixed shapes: the padded batches all have the same
            self.compiled_model = CompiledPotential(model, dynamic=False)
            # the backward pass is part of the graph; the task steps the optimizer
            self.automatic_optimization = False
        elif compile and self.distillation is not None:
            warnings.warn(
                "distillation runs the model without compilation.", stacklevel=2
            )
        elif compile:
            self.compiled_model = CompiledPotential(model)

        # 1.0 / 0.0 whether this step updated the parameters; the EMA callback
        # skips its update with the optimizer step. None: always updated.
        self.keep_step: torch.Tensor | None = None
        self.skipped_steps: torch.Tensor | None = None
        self._graph = None
        # cuda_graph: the training loss summed on the GPU, read once per validation
        self._loss_sum: torch.Tensor | None = None
        self._loss_steps = 0
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            if self.distillation is None:
                self.model.initialize_transforms(self.trainer.datamodule)
            else:
                self.distillation.setup(self.trainer.datamodule)
            if self.cuda_graph:
                dm = self.trainer.datamodule
                if getattr(dm, "data_device", None) is None:
                    raise ValueError(
                        "cuda_graph needs the data on the GPU: "
                        "AtomsDataModule(data_device='cuda')"
                    )
                dm.train_batches_in_step = True  # the graph builds the batch

    def forward(self, inputs: dict[str, torch.Tensor]):
        if self.compiled_model is not None:
            return self.compiled_model(inputs)
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

    def log_padded_metrics(self, pred, targets, batch, subset):
        """The metrics of a padded batch, over its real rows. Finding them is a host
        sync, which drains the queue of GPU work; it is done once per padding mask
        (atoms, structures) and shared by the outputs."""
        rows: dict[int, torch.Tensor] = {}
        for output in self.outputs:
            if not output.metrics[subset]:
                continue
            mask = padding_mask(pred[output.name], batch)
            if id(mask) not in rows:
                rows[id(mask)] = mask.nonzero().squeeze(-1)
            p = pred[output.name].detach().index_select(0, rows[id(mask)])
            t = targets[output.target_property].index_select(0, rows[id(mask)])
            for metric_name, metric in output.metrics[subset].items():
                metric(p, t)
                self.log(
                    f"{subset}_{output.target_property}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def _step(self, batch, subset):
        """Loss of one batch, summed over the outputs; logs the metrics of
        ``subset``.

        Without distillation, the model is compared with the batch's labels.
        With distillation, ``Distillation.step`` also adds the teacher's
        targets and the student's curvature, if an output asks for them. Outside
        training, the curvature probes come from a seeded generator, so every
        validation epoch uses the same probes. A padded batch (data on the GPU) is
        compared on its real rows only."""
        padded = properties.atom_mask in batch
        if padded and self.distillation is not None:
            raise ValueError("distillation takes no padded batches (data_device)")
        if self.distillation is None:
            targets = extract_targets(self.outputs, batch)
            model = self.compiled_model or self.model
            pred = predict_without_postprocessing(model, batch)
        else:
            pred, targets = self.distillation.step(batch, train=subset == "train")
        if padded:
            loss = calculate_padded_loss(self.outputs, pred, targets, batch)
        else:
            loss = calculate_loss(self.outputs, pred, targets)
        if subset != "train" or self.writes_train_logs():
            if padded:
                self.log_padded_metrics(pred, targets, batch, subset)
            else:
                self.log_metrics(pred, targets, subset)
        return loss

    def writes_train_logs(self) -> bool:
        """Whether Lightning writes this training step's logs (every
        ``log_every_n_steps``). Training metrics are computed and logged only then:
        the values of the other steps are never written, and computing them cost a
        quarter of a fast training step. Without a trainer, always."""
        trainer = self._trainer
        if trainer is None:
            return True
        # Lightning's own condition (update_train_step_metrics)
        return trainer._logger_connector.should_update_logs or bool(
            trainer.fast_dev_run
        )

    def on_validation_epoch_start(self):
        if self.distillation is not None:
            self.distillation.reset_probes()
        if self._loss_steps:  # cuda_graph: the mean training loss (one host sync)
            self.log(
                "train_loss_epoch",
                float(self._loss_sum) / self._loss_steps,
                on_epoch=True,
            )
            self._loss_sum, self._loss_steps = None, 0

    def on_test_epoch_start(self):
        if self.distillation is not None:
            self.distillation.reset_probes()

    def training_step(self, batch, batch_idx):
        if self.cuda_graph:
            return self._graph_training_step(batch)
        loss = self._step(batch, "train")
        if self.writes_train_logs():
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        if self.skip_loss_above is not None:
            keep = bool(loss.detach() <= self.skip_loss_above)  # a host sync
            self._count_step(torch.tensor(0.0 if keep else 1.0, device=loss.device))
            if not keep:
                return None  # Lightning skips the optimizer step
        return loss

    def _count_step(self, skip: torch.Tensor):
        self.keep_step = 1.0 - skip
        self.skipped_steps = (
            skip if self.skipped_steps is None else self.skipped_steps + skip
        )

    def _batch_size(self, batch) -> int:
        if properties.structure_mask in batch:  # a padded batch: its real structures
            return int(batch[properties.structure_mask].sum())
        return len(batch[properties.idx])

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        loss = self._step(batch, "val")
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self._batch_size(batch),
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
            batch_size=self._batch_size(batch),
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

    def _warmup(self, param_groups):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
            for pg in param_groups:
                pg["lr"] = lr_scale * self.lr

    def optimizer_step(
        self,
        epoch: int | None = None,
        batch_idx: int | None = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        self._warmup(optimizer.param_groups)

        # update params
        optimizer.step(closure=optimizer_closure)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        if self.grad_clip is not None:
            gradient_clip_val, gradient_clip_algorithm = self.grad_clip, "norm"
        self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    # -- the training step as one CUDA graph (cuda_graph) ---------------------- #

    def _record_graph(self, sel: torch.Tensor):
        """Record building the batch, the forward pass, the loss and the backward
        pass as one CUDA graph, whose only input are the structure indices."""
        dm = self.trainer.datamodule
        self._graph_input = sel.clone()

        def step():
            batch = dm.build_batch(self._graph_input, "train")
            targets = extract_targets(self.outputs, batch)
            pred = predict_without_postprocessing(self.compiled_model, batch)
            loss = calculate_padded_loss(self.outputs, pred, targets, batch)
            loss.backward()
            return loss, pred, targets, batch

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # compiles here, outside the capture
                step()
        torch.cuda.current_stream().wait_stream(side)
        for p in self.model.parameters():
            p.grad = None
        self._graph = torch.cuda.CUDAGraph()
        # the AccumulateGrad nodes of the parameters remember the stream of an
        # earlier backward; let the capturing stream take over
        if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
            torch.autograd.graph.set_override_stale_capture_stream(True)
        with torch.cuda.graph(self._graph, stream=side):
            self._graph_output = step()

    def _graph_training_step(self, sel: torch.Tensor):
        """Replay the graph with new structure indices, then clip, step the
        optimizer and decide on skipping, all without a host sync."""
        if self._graph is None:
            self._record_graph(sel)
        self._graph_input.copy_(sel, non_blocking=True)
        self._graph.replay()
        loss, pred, targets, batch = self._graph_output
        opt = self.optimizers()
        self._warmup(opt.param_groups)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if self.skip_loss_above is None:
            opt.step()
        elif opt.optimizer.defaults.get("fused", False):
            # on the GPU: the fused step skips itself where found_inf is 1
            skip = (~(loss.detach() <= self.skip_loss_above)).to(torch.float32)
            opt.optimizer.found_inf = skip
            self._count_step(skip)
            opt.step()
        else:  # a host sync every step
            keep = bool(loss.detach() <= self.skip_loss_above)
            self._count_step(torch.tensor(0.0 if keep else 1.0, device=loss.device))
            if keep:
                opt.step()

        self._loss_sum = (
            loss.detach().clone()
            if self._loss_sum is None
            else self._loss_sum.add_(loss.detach())
        )
        self._loss_steps += 1
        if self.writes_train_logs():
            self.log_padded_metrics(pred, targets, batch, "train")
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss.detach()

    def on_validation_model_zero_grad(self):
        # Lightning sets every .grad to None before validation by default. The
        # graph writes the gradients into its own tensors, which would then no
        # longer be the parameters' .grad, and the optimizer would skip them.
        if not self.cuda_graph:
            super().on_validation_model_zero_grad()

    def on_train_start(self):
        if self.cuda_graph:
            # Python's garbage collector walks every tracked object in a full
            # collection; with the model, data and compiled graphs that is a stall
            # of milliseconds, in which the GPU runs dry. Objects alive now are
            # left out.
            gc.collect()
            gc.freeze()

    def on_train_end(self):
        if self.cuda_graph:
            gc.unfreeze()

    def _schedulers(self) -> list:
        sch = self.lr_schedulers()
        if sch is None:
            return []
        return sch if isinstance(sch, list) else [sch]

    def on_validation_epoch_end(self):
        # with cuda_graph (manual optimization) the task steps the schedulers:
        # ReduceLROnPlateau after validation, the others after each epoch
        if not self.cuda_graph or self.trainer.sanity_checking:
            return
        for sch in self._schedulers():
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = self.trainer.callback_metrics.get(self.schedule_monitor)
                if metric is not None:
                    sch.step(metric)

    def on_train_epoch_end(self):
        if not self.cuda_graph:
            return
        for sch in self._schedulers():
            if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step()

    def save_model(self, path: str, do_postprocessing: bool | None = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            try:
                torch.save(self.model, path)
            finally:
                self.model.do_postprocessing = pp_status

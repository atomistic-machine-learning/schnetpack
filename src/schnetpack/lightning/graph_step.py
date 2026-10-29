"""
A training task whose whole step is one CUDA graph.

With the data on the GPU (:class:`~schnetpack.lightning.DeviceAtomsDataModule`), a
step can build its own batch from structure indices. :class:`GraphStepTask` records
building the batch (padding, transforms such as noise), the compiled forward pass,
the loss and the backward pass once as a CUDA graph and replays it every step with
new indices: one launch instead of hundreds of kernels and no host synchronization,
so the GPU never waits for Python. The optimizer step (fused AdamW recommended),
gradient clipping, skipping of exploding steps and the EMA run after the replay,
also without a host sync.

On QM9 (PaiNN, energy, batch 100) this took a training step from 6.0 ms
(compiled, fused AdamW, foreach EMA) to 2.7 ms, and GPFF's (PaiNN 4 x 256, batch
512) from 158 ms (the library's DataLoader route) to 30 ms with the banded filter;
see ``docs_new/speedup/step_cost_log.md``.
"""

from typing import Any

import torch

from schnetpack import properties
from schnetpack.generative.outputs import VariableOutput
from schnetpack.lightning.task import AtomisticTask
from schnetpack.model.compile import CompiledPotential
from schnetpack.objectives import (
    ModelOutput,
    UnsupervisedModelOutput,
    extract_targets,
    predict_without_postprocessing,
)

__all__ = ["GraphStepTask", "padded_loss"]


def _row_mask(p: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The padding mask (1.0 real, 0.0 padding) that matches an output's rows."""
    atom_mask = batch[properties.atom_mask]
    if p.shape[0] == atom_mask.shape[0]:
        return atom_mask
    return batch[properties.structure_mask]


def padded_loss(
    outputs: list[ModelOutput],
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    The composite loss of a padded batch over its real rows, with fixed shapes and
    no host sync (recordable in a CUDA graph).

    :class:`~schnetpack.generative.VariableOutput` handles padding itself (weight 0
    on padding rows). Other outputs need an ``MSELoss`` or ``L1Loss`` with mean
    reduction and no masks: the mean is taken over the rows the padding mask keeps,
    as the loss over those rows alone would give.
    """
    loss: torch.Tensor | float = 0.0
    for o in outputs:
        if o.loss_weight == 0:
            continue
        if isinstance(o, VariableOutput):
            loss = loss + o.calculate_loss(pred, targets)
            continue
        p, t = pred[o.name], targets[o.target_property]
        w = _row_mask(p, batch).to(p.dtype).view(-1, *[1] * (p.dim() - 1))
        if isinstance(o.loss_fn, torch.nn.MSELoss):
            err = (p - t) ** 2
        else:  # L1Loss, checked by GraphStepTask
            err = (p - t).abs()
        loss = loss + o.loss_weight * (w * err).sum() / (w.sum() * p[0].numel())
    return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss)


class GraphStepTask(AtomisticTask):
    """
    :class:`~schnetpack.lightning.AtomisticTask` with the training step as one CUDA
    graph (module docstring). Needs a datamodule that builds batches from structure
    indices (:class:`~schnetpack.lightning.DeviceAtomsDataModule`) and a CUDA device.

    The model always runs compiled, with static shapes. The training step uses
    manual optimization: Lightning's ``gradient_clip_val`` does not apply (use
    ``grad_clip``), the learning-rate warmup and the schedulers are stepped here
    (a ``ReduceLROnPlateau`` after each validation, with the monitored metric;
    others after each training epoch). Training metrics are computed on logging
    steps only. Validation and testing run the compiled model on padded batches
    without a graph; ``val_loss`` is the mean over the real atoms when an output is
    per atom, else over the real structures.

    Supported outputs: :class:`~schnetpack.generative.VariableOutput`, and
    :class:`~schnetpack.objectives.ModelOutput` with ``MSELoss`` or ``L1Loss`` and
    no loss masks. No knowledge distillation.
    """

    def __init__(
        self,
        *args,
        grad_clip: float | None = None,
        skip_loss_above: float | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            *args: as :class:`~schnetpack.lightning.AtomisticTask`
            grad_clip: clip the gradient norm to this value
            skip_loss_above: skip the optimizer step (and the EMA update) when the
                loss exceeds this value or is not finite. Without a host sync for a
                fused optimizer (it reads the flag as ``found_inf``); otherwise the
                loss is read on the host every step.
            **kwargs: as :class:`~schnetpack.lightning.AtomisticTask`; ``compile``
                is implied
        """
        kwargs["compile"] = False  # compiled below, with static shapes
        super().__init__(*args, **kwargs)
        if self.distillation is not None:
            raise ValueError("GraphStepTask does not support knowledge distillation")
        for o in self.outputs:
            if isinstance(o, VariableOutput) or o.loss_weight == 0:
                continue
            if isinstance(o, UnsupervisedModelOutput) or o.masks:
                raise ValueError(f"GraphStepTask: output {o.name!r} is not supported")
            if not (
                isinstance(o.loss_fn, (torch.nn.MSELoss, torch.nn.L1Loss))
                and o.loss_fn.reduction == "mean"
            ):
                raise ValueError(
                    f"GraphStepTask: output {o.name!r} needs MSELoss or L1Loss with "
                    f"mean reduction, got {o.loss_fn}"
                )
        self.automatic_optimization = False
        self.compiled_model = CompiledPotential(self.model, dynamic=False)
        self.grad_clip, self.skip_loss_above = grad_clip, skip_loss_above
        self.graph = None
        self.keep_step: torch.Tensor | None = None  # 1.0 / 0.0, read by the EMA
        self.skipped_steps: torch.Tensor | None = None
        # the training loss summed on the GPU, read once per validation
        self._loss_sum: torch.Tensor | None = None
        self._loss_steps = 0
        self.train_loss_mean = float("nan")

    def setup(self, stage=None):
        super().setup(stage)
        dm = self.trainer.datamodule
        if not hasattr(dm, "build_batch"):
            raise TypeError(
                "GraphStepTask needs a datamodule that builds batches from structure "
                "indices, e.g. DeviceAtomsDataModule"
            )
        dm.train_batches_in_step = True

    def on_validation_model_zero_grad(self):
        # Lightning sets every .grad to None before validation by default. The
        # graph writes the gradients into its own tensors, which would then no
        # longer be the parameters' .grad, and the optimizer would skip them.
        pass

    # -- the graph -------------------------------------------------------- #

    def _capture(self, sel: torch.Tensor):
        dm = self.trainer.datamodule
        self.static_sel = sel.clone()

        def step():
            batch = dm.build_batch(self.static_sel, "train")
            targets = extract_targets(self.outputs, batch)
            pred = predict_without_postprocessing(self.compiled_model, batch)
            loss = padded_loss(self.outputs, pred, targets, batch)
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
        self.graph = torch.cuda.CUDAGraph()
        # the AccumulateGrad nodes of the parameters remember the stream of an
        # earlier backward; let the capturing stream take over
        if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
            torch.autograd.graph.set_override_stale_capture_stream(True)
        with torch.cuda.graph(self.graph, stream=side):
            self.static_out = step()

    def training_step(self, sel, batch_idx):
        if self.graph is None:
            self._capture(sel)
        self.static_sel.copy_(sel, non_blocking=True)
        self.graph.replay()
        loss, pred, targets, batch = self.static_out
        opt = self.optimizers()

        if self.global_step < self.warmup_steps:
            scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
            for group in opt.param_groups:
                group["lr"] = scale * self.lr
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if self._keep(loss, opt.optimizer):
            opt.step()

        self._loss_sum = (
            loss.detach().clone()
            if self._loss_sum is None
            else self._loss_sum.add_(loss.detach())
        )
        self._loss_steps += 1
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self._log_padded_metrics(pred, targets, batch, "train")
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss.detach()

    def _keep(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> bool:
        """Whether to run the optimizer step; sets keep_step for the EMA."""
        if self.skip_loss_above is None:
            self.keep_step = None
            return True
        if optimizer.defaults.get("fused", False):
            # on the GPU: the fused step skips itself where found_inf is 1
            skip = (~(loss.detach() <= self.skip_loss_above)).to(torch.float32)
            optimizer.found_inf = skip
            keep = True
        else:  # a host sync every step
            keep = bool(loss.detach() <= self.skip_loss_above)
            skip = torch.tensor(0.0 if keep else 1.0, device=loss.device)
        self.keep_step = 1.0 - skip
        self.skipped_steps = (
            skip if self.skipped_steps is None else self.skipped_steps + skip
        )
        return keep

    def on_validation_epoch_start(self):
        """Logs ``train_loss_epoch``: the mean training loss since the last
        validation (one host sync)."""
        super().on_validation_epoch_start()
        if self._loss_steps:
            self.train_loss_mean = float(self._loss_sum) / self._loss_steps
            self._loss_sum, self._loss_steps = None, 0
            self.log("train_loss_epoch", self.train_loss_mean, on_epoch=True)

    # -- schedulers (manual optimization) ---------------------------------- #

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        for sch in self._schedulers():
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = self.trainer.callback_metrics.get(self.schedule_monitor)
                if metric is not None:
                    sch.step(metric)

    def on_train_epoch_end(self):
        for sch in self._schedulers():
            if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step()

    def _schedulers(self) -> list:
        sch = self.lr_schedulers()
        if sch is None:
            return []
        return sch if isinstance(sch, list) else [sch]

    # -- validation and testing ------------------------------------------- #

    def _log_padded_metrics(self, pred, targets, batch, subset):
        """Metrics over the real rows (boolean selection: a host sync)."""
        for o in self.outputs:
            if not o.metrics[subset]:
                continue
            keep = _row_mask(pred[o.name], batch).bool()
            p = pred[o.name][keep].detach()
            t = targets[o.target_property][keep]
            for name, metric in o.metrics[subset].items():
                metric(p, t)
                self.log(
                    f"{subset}_{o.target_property}_{name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def _eval_step(self, batch, subset):
        torch.set_grad_enabled(self.grad_enabled)
        targets = extract_targets(self.outputs, batch)
        pred = predict_without_postprocessing(self.compiled_model, batch)
        loss = padded_loss(self.outputs, pred, targets, batch)
        self._log_padded_metrics(pred, targets, batch, subset)
        per_atom = any(
            pred[o.name].shape[0] == batch[properties.atom_mask].shape[0]
            for o in self.outputs
        )
        mask = batch[properties.atom_mask if per_atom else properties.structure_mask]
        self.log(
            f"{subset}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(mask.sum()),
        )
        return {f"{subset}_loss": loss}

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

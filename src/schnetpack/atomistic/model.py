from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List, Type, TYPE_CHECKING

from schnetpack.transform import Transform

import torch
import pytorch_lightning as pl
import torch.nn as nn

import schnetpack as spk
from torchmetrics import Metric

__all__ = ["AtomisticModel", "ModelOutput"]


class ModelOutput(nn.Module):
    """
    Defines an output for the model, including optional loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        target_name: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_name: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of
        """
        super().__init__()
        self.property = name
        self.target_name = target_name or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = nn.ModuleDict(metrics)


class AtomisticModel(pl.LightningModule):
    """
    Base class for all SchNetPack models.

    """

    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: nn.Module,
        output_modules: List[nn.Module],
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        postprocess: Optional[List[Transform]] = None,
    ):
        """
        Args:
            datamodule: pytorch_lightning module for dataset
            representation: nn.Module for atomistic representation
            output_modules: List of module to compute outputs from atomistic representation.
                Output modules must modify and return the input dictionary.
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            postprocess: list of postprocessors to be applied to model for predictions
        """
        super().__init__()
        self.datamodule = datamodule
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor

        # self.save_hyperparameters(
        #     "representation", "output_modules", "postprocess", "outputs"
        # )
        self.representation = representation
        self.outputs = nn.ModuleList(outputs)
        self.output_modules = nn.ModuleList(output_modules)
        self.pp = postprocess or []

        self.required_derivatives = []  # set()
        for m in self.output_modules:
            if hasattr(m, "required_derivatives"):
                for p in m.required_derivatives:
                    if p not in self.required_derivatives:
                        self.required_derivatives.append(p)

        self.grad_enabled = len(self.required_derivatives) > 0
        self.inference_mode = False

    def setup(self, stage=None):
        self.postprocessors = torch.nn.ModuleList()
        for pp in self.pp:
            pp.postprocessor()
            pp.datamodule(self.datamodule)
            self.postprocessors.append(pp)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        for p in self.required_derivatives:
            inputs[p].requires_grad_()

        inputs.update(self.representation(inputs))

        for outmod in self.output_modules:
            inputs.update(outmod(inputs))

        results = {out.property: inputs[out.property] for out in self.outputs}
        results = self.postprocess(inputs, results)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss_p = output.loss_weight * output.loss_fn(
                pred[output.property], batch[output.target_name]
            )
            loss += loss_p
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            for metric_name, pmetric in output.metrics.items():
                self.log(
                    f"{subset}_{output.property}_{metric_name}",
                    pmetric(pred[output.property], targets[output.target_name]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def training_step(self, batch, batch_idx):
        targets = {
            output.target_name: batch[output.target_name] for output in self.outputs
        }
        pred = self(batch)
        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(targets, pred, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        targets = {
            output.target_name: batch[output.target_name] for output in self.outputs
        }
        pred = self(batch)
        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics(targets, pred, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        targets = {
            output.target_name: batch[output.target_name] for output in self.outputs
        }
        pred = self(batch)
        loss = self.loss_fn(pred, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics(targets, pred, "test")
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)

            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            return [optimizer], [optimconf]
        else:
            return optimizer

    def postprocess(
        self, inputs: Dict[str, torch.Tensor], results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.inference_mode:
            for pp in self.postprocessors:
                results = pp(inputs, results)
        return results

    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[torch.ScriptModule, Dict[str, torch.ScriptModule]]:
        imode = self.inference_mode
        self.inference_mode = True
        script = super().to_torchscript(file_path, method, example_inputs, **kwargs)
        self.inference_mode = imode
        return script

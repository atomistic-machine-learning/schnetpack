import logging

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable

import schnetpack as spk
from schnetpack.model.base import AtomisticModel

log = logging.getLogger(__name__)

__all__ = ["SinglePropertyModel"]


class SinglePropertyModel(AtomisticModel):
    """
    AtomisticModel for models that predict single chemical properties, e.g. for QM9 benchmarks.

    """

    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: nn.Module,
        output: nn.Module,
        target_property: str = "energy",
        loss_fn: Optional[Callable] = None,
        metrics: Optional[List[Dict]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
        scheduler_monitor: Optional[str] = None,
        postprocess: Optional[List[spk.transform.Transform]] = None,
    ):
        """
        Args:
            datamodule: pytorch_lightning module for dataset
            representation: nn.Module for atomistic representation
            output: nn.Module for computation of physical properties from atomistic representation
            target_property: name of targeted property in datamodule, e.g. energy/gap/...
            loss_fn: function for computation of loss
            metrics: dict of metrics for predictions of the target property with metric name as keys and callable as values
            optimizer: functools.partial-function with torch optimizer and args for creation of optimizer with
                optimizer(params=self.parameters())
            scheduler: functools.partial-function with scheduler and args for creation of scheduler with
                scheduler(optimizer=optimizer)
            scheduler_monitor: name of metric to be observed and used for lr drops of scheduler
            postprocess: list of postprocessors to be applied to model for predictions
        """
        super(SinglePropertyModel, self).__init__(
            datamodule=datamodule,
            representation=representation,
            output=output,
            postprocess=postprocess,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_monitor=scheduler_monitor,
        )

        self.target_property = target_property
        self.loss_fn = loss_fn

        self.metrics = nn.ModuleDict(
            {target_property + "_" + name: metric for name, metric in metrics.items()}
        )

    def forward(self, inputs):
        inputs.update(self.representation(inputs))
        pred = self.output(inputs)

        # todo: missing postprocessor?
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.target_property]
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for name, metric in self.metrics.items():
            self.log(
                f"train_{name}",
                metric(pred, target),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.target_property]
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log(
                f"val_{name}",
                metric(pred, target),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch[self.target_property]
        loss = self.loss_fn(pred, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log(
                f"test_{name}",
                metric(pred, target),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return {"test_loss": loss}

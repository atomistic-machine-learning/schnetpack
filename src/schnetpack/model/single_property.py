import logging

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Any, Type

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
        target_property: str,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[List[Dict]] = None,
        scheduler_cls: Type = None,
        scheduler_args: Dict[str, Any] = None,
        scheduler_monitor: Optional[str] = None,
        postprocess: Optional[List[spk.transform.Transform]] = None,
    ):
        """
        Args:
            datamodule: pytorch_lightning module for dataset
            representation: nn.Module for atomistic representation
            output: nn.Module for computation of physical properties from atomistic representation
            target_property: name of targeted property in datamodule
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            loss_fn: function for computation of loss
            metrics: dict of metrics for predictions of the target property with metric name as keys and callable as values
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            postprocess: list of postprocessors to be applied to model for predictions
        """
        super(SinglePropertyModel, self).__init__(
            datamodule=datamodule,
            representation=representation,
            output=output,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            postprocess=postprocess,
        )

        self.target_property = target_property
        self.loss_fn = loss_fn

        self.metrics = nn.ModuleDict(
            {target_property + "_" + name: metric for name, metric in metrics.items()}
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        inputs.update(self.representation(inputs))
        results = {self.target_property: self.output(inputs)}
        results = self.postprocess(inputs, results)
        return results

    def training_step(self, batch, batch_idx):
        results = self(batch)
        pred = results[self.target_property]
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
        results = self(batch)
        pred = results[self.target_property]
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
        results = self(batch)
        pred = results[self.target_property]
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

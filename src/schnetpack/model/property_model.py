import logging

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Any, Type, Union

import schnetpack as spk
from schnetpack.model.base import AtomisticModel
from omegaconf.listconfig import ListConfig

log = logging.getLogger(__name__)

__all__ = ["PropertyModel"]


class PropertyModel(AtomisticModel):
    """
    AtomisticModel for models that predict single chemical properties, e.g. for QM9 benchmarks.

    This Module can be used for single property prediction, as well as for the prediction of multiple properties.
    For multi-property prediction the following arguments need to be passed as list: 'properties', 'targets',
    'loss_fn', 'loss_weights' and 'metrics'. The length of these lists must match in order to find the correct
    mapping of the list items by id!
    """

    def __init__(
        self,
        datamodule: spk.data.AtomsDataModule,
        representation: nn.Module,
        output: Union[nn.Module, List[nn.Module]],
        properties: Union[str, List[str]],
        targets: Union[str, List[str]],
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_args: Optional[Dict[str, Any]] = None,
        loss_fn: Union[Dict[str, Callable], List[Dict[str, Callable]]] = None,
        loss_weights: Optional[Union[float, List[float]]] = None,
        metrics: Optional[Union[Dict[str, Callable], List[Dict[str, Callable]]]] = None,
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
            properties: name of property in results dict
            targets: name of properties in datamodule
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            loss_fn: function for computation of loss
            loss_weights: list of loss weights for multi property training
            metrics: dict of metrics for predictions of the target property with metric name as keys and callable as values
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            postprocess: list of postprocessors to be applied to model for predictions
        """
        # change to lists if single property
        if type(properties) == str:
            properties = [properties]
        if type(targets) == str:
            targets = [targets]
        if type(loss_fn) not in [list, ListConfig]:
            loss_fn = [loss_fn]
        if loss_weights is None:
            loss_weights = [1.0 for _ in properties]
        if type(loss_weights) not in [list, ListConfig]:
            loss_weights = [loss_weights]
        if type(output) not in [list, ListConfig]:
            output = [output]
        if type(metrics) not in [list, ListConfig]:
            metrics = [metrics]

        super(PropertyModel, self).__init__(
            datamodule=datamodule,
            representation=representation,
            output=nn.ModuleDict({p: o for p, o in zip(properties, output)}),
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            postprocess=postprocess,
        )

        self.properties = properties
        self.targets = {p: t for p, t in zip(properties, targets)}

        self.losses = nn.ModuleDict({p: loss for p, loss in zip(properties, loss_fn)})
        self.loss_weights = {p: w for p, w in zip(properties, loss_weights)}

        self.metrics = nn.ModuleDict()
        for pname, pmetrics in zip(properties, metrics):
            self.metrics.update(
                {pname + ":" + name: metric for name, metric in pmetrics.items()}
            )

    def loss_fn(self, pred, batch):
        loss = 0.0
        for pname in self.properties:
            loss_p = self.loss_weights[pname] * self.losses[pname](
                pred[pname], batch[self.targets[pname]]
            )
            loss += loss_p
        return loss

    def forward(self, inputs: Dict[str, torch.Tensor]):
        inputs.update(self.representation(inputs))
        results = {}
        for pname, output in self.output.items():
            results[pname] = output(inputs)
        results = self.postprocess(inputs, results)
        return results

    def log_metrics(self, pred, batch, subset):
        for metric_name, pmetric in self.metrics.items():
            prop, name = metric_name.split(":")
            self.log(
                f"{subset}_{metric_name}".replace(":", "_"),
                pmetric(pred[prop], batch[self.targets[prop]]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics(pred, batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, batch, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, batch, "test")

        return {"test_loss": loss}

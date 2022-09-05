from pathlib import Path
from typing import Optional, Dict, List, Type, Any, Union

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from torch.autograd import grad

from schnetpack.model.base import AtomisticModel

__all__ = ["ModelOutput", "AtomisticTask"]


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = nn.ModuleDict(metrics)
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def calculate_metrics(self, pred, target):
        metrics = {
            metric_name: metric(pred[self.name], target[self.target_property])
            for metric_name, metric in self.metrics.items()
        }
        return metrics


class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """

    def calculate_loss(self, pred, target=None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])
        return loss

    def calculate_metrics(self, pred, target=None):
        metrics = {
            metric_name: metric(pred[self.name])
            for metric_name, metric in self.metrics.items()
        }
        return metrics


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
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
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            for metric_name, metric in output.calculate_metrics(pred, targets).items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = None,
        optimizer_closure=None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ):
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.trainer is None or self.trainer.strategy.local_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing

            torch.save(self.model, path)

            self.model.do_postprocessing = pp_status


class ConsiderOnlySelectedAtoms(nn.Module):
    """
    Constraint that allows to neglect some atomic targets (e.g. forces of some specified atoms) for model optimization,
    while not affecting the actual model output. The indices of the atoms, which targets to consider in the loss
    function, must be provided in the dataset for each sample in form of a torch tensor of type boolean
    (True: considered, False: neglected).
    """

    def __init__(self, selection_name):
        """
        Args:
            selection_name: string associated with the list of considered atoms in the dataset
        """
        super().__init__()
        self.selection_name = selection_name

    def forward(self, pred, targets, output_module):
        """
        A torch tensor is loaded from the dataset, which specifies the considered atoms. Only the
        predictions of those atoms are considered for training, validation, and testing.

        :param pred: python dictionary containing model outputs
        :param targets: python dictionary containing targets
        :param output_module: torch.nn.Module class of a particular property (e.g. forces)
        :return: model outputs and targets of considered atoms only
        """

        considered_atoms = targets[self.selection_name].nonzero()[:, 0]

        # drop neglected atoms
        pred[output_module.name] = pred[output_module.name][considered_atoms]
        targets[output_module.target_property] = targets[output_module.target_property][
            considered_atoms
        ]

        return pred, targets

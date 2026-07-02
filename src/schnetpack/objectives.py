"""
Loss assembly for training — pure PyTorch, no Lightning dependency.

``ModelOutput`` maps a model output to a loss function, loss weight and metrics.
The module-level functions assemble the composite loss from a list of outputs,
so a model can be trained in a hand-written PyTorch loop:

    loss = compute_loss(outputs, model, batch)
    loss.backward()

:class:`schnetpack.AtomisticTask` uses these same functions for training with
the PyTorch Lightning Trainer.
"""
from typing import Optional, Dict, List

import torch
from torch import nn as nn
from torchmetrics import Metric

__all__ = [
    "ModelOutput",
    "UnsupervisedModelOutput",
    "ConsiderOnlySelectedAtoms",
    "extract_targets",
    "apply_constraints",
    "calculate_loss",
    "compute_loss",
]


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
        r"""
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
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])


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

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name])


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


def extract_targets(
    outputs: List[ModelOutput], batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Collect the target properties of all supervised outputs from a batch."""
    targets = {
        output.target_property: batch[output.target_property]
        for output in outputs
        if not isinstance(output, UnsupervisedModelOutput)
    }
    if "considered_atoms" in batch:
        targets["considered_atoms"] = batch["considered_atoms"]
    return targets


def apply_constraints(
    outputs: List[ModelOutput],
    pred: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
):
    for output in outputs:
        for constraint in output.constraints:
            pred, targets = constraint(pred, targets, output)
    return pred, targets


def calculate_loss(
    outputs: List[ModelOutput],
    pred: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Weighted composite loss over all outputs."""
    loss = 0.0
    for output in outputs:
        loss = loss + output.calculate_loss(pred, targets)
    return loss


def compute_loss(
    outputs: List[ModelOutput],
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    One-call loss for hand-written training loops: extract targets, run the
    model, apply constraints and assemble the weighted composite loss.

    Takes the model rather than precomputed predictions because SchNetPack
    models write their results into the input dict — the targets must be
    extracted from the batch before the forward pass overwrites them.
    """
    targets = extract_targets(outputs, batch)
    pred = model(batch)
    pred, targets = apply_constraints(outputs, pred, targets)
    return calculate_loss(outputs, pred, targets)

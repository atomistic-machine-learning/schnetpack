"""
Loss assembly for training — pure PyTorch, no Lightning dependency.

``ModelOutput`` maps a model output to a loss function, loss weight and metrics.
The module-level functions assemble the composite loss from a list of outputs,
so a model can be trained in a hand-written PyTorch loop:

    targets = extract_targets(outputs, batch)
    pred = model(batch)
    loss = calculate_loss(outputs, pred, targets)
    loss.backward()

Batches padded to fixed shapes (data held on the GPU,
:class:`~schnetpack.lightning.AtomsDataModule` with ``data_device``) mark their
padding rows in ``properties.atom_mask`` / ``properties.structure_mask``;
:func:`calculate_padded_loss` leaves them out without a host sync.

:class:`schnetpack.lightning.AtomisticTask` uses these same functions for training with
the PyTorch Lightning Trainer.
"""

from collections.abc import Collection

import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack import properties
from schnetpack.model.base import AtomisticModel

__all__ = [
    "ModelOutput",
    "UnsupervisedModelOutput",
    "LossMask",
    "AtomMask",
    "extract_targets",
    "predict_without_postprocessing",
    "calculate_loss",
    "calculate_padded_loss",
    "padding_mask",
]


class LossMask(nn.Module):
    """
    Restricts which entries of a model output are compared with its target.

    A mask returns one bool per entry along the first dimension of the
    prediction (per atom or per structure); only the entries it keeps enter
    the loss and the logged metrics of its :class:`ModelOutput`. The masks of
    one output combine by logical AND. A mask never changes the model output
    itself.

    Subclasses implement :meth:`forward` and name the batch keys it reads in
    :attr:`required_keys`, so that :func:`extract_targets` collects them.
    """

    @property
    def required_keys(self) -> tuple[str, ...]:
        """The batch keys :meth:`forward` reads from the targets."""
        return ()

    def forward(self, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class AtomMask(LossMask):
    """
    Keeps the entries flagged in a batch key, e.g. to leave the forces of some
    atoms out of training. The dataset stores one flag per atom (True:
    considered, False: neglected).
    """

    def __init__(self, key: str):
        """
        Args:
            key: batch key of the per-atom flags.
        """
        super().__init__()
        self.key = key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def forward(self, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        return targets[self.key].bool()


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: nn.Module | None = None,
        loss_weight: float = 1.0,
        metrics: dict[str, Metric] | None = None,
        masks: list[LossMask] | None = None,
        target_property: str | None = None,
    ):
        r"""
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            masks: loss masks restricting which entries of the output are
                compared with the target, in the loss and in the metrics, e.g.
                to neglect the forces of some atoms. They don't change the model
                output; see :class:`LossMask`.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        metrics = metrics or {}
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.masks = nn.ModuleList(masks or [])

    def masked(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This output's prediction and target, restricted to the entries its
        masks keep."""
        prediction, reference = pred[self.name], target[self.target_property]
        if not self.masks:
            return prediction, reference
        keep = self.masks[0](target)
        for mask in self.masks[1:]:
            keep = keep & mask(target)
        if keep.shape != prediction.shape[:1]:
            raise ValueError(
                f"the masks of output {self.name!r} select from {keep.shape[0]} "
                f"entries, but its prediction has {prediction.shape[0]}"
            )
        return prediction[keep], reference[keep]

    @property
    def loss_keys(self) -> tuple[str, ...]:
        """Batch keys the loss reads besides the target (e.g. a per-row weight);
        :func:`extract_targets` copies those the batch has."""
        return ()

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(*self.masked(pred, target))
        return loss

    def calculate_padded_loss(self, pred, target, mask):
        """
        The loss over the rows where ``mask`` is 1 (a padded batch), as the loss over
        those rows alone, without a host sync. Needs ``MSELoss`` or ``L1Loss`` with
        mean reduction and no loss masks.
        """
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        mse = isinstance(self.loss_fn, nn.MSELoss)
        if self.masks or not (mse or isinstance(self.loss_fn, nn.L1Loss)):
            raise ValueError(
                f"output {self.name!r}: padded batches need MSELoss or L1Loss "
                "without loss masks"
            )
        if self.loss_fn.reduction != "mean":
            raise ValueError(f"output {self.name!r}: padded batches need mean loss")
        p, t = pred[self.name], target[self.target_property]
        err = (p - t) ** 2 if mse else (p - t).abs()
        w = mask.to(p.dtype).view(-1, *[1] * (p.dim() - 1))
        return self.loss_weight * (w * err).sum() / (w.sum() * p[0].numel())

    def update_metrics(self, pred, target, subset):
        prediction, reference = self.masked(pred, target)
        for metric in self.metrics[subset].values():
            metric(prediction, reference)


class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged. It takes no masks, as it
    has no target.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.masks:
            raise ValueError(
                f"unsupervised output {self.name!r} has no target, so it takes no masks"
            )

    def calculate_loss(self, pred, target=None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name])


def extract_targets(
    outputs: list[ModelOutput],
    batch: dict[str, torch.Tensor],
    skip: Collection[str] = (),
) -> dict[str, torch.Tensor]:
    """Collect the target properties of all supervised outputs from a batch.

    Targets named in ``skip`` are left out: something other than the batch
    supplies them, e.g. the teacher in knowledge distillation. The keys the
    outputs' masks read are collected too, whoever supplies the target. Call it
    before the forward pass: SchNetPack models write their results into the
    input dict, so a target stored under an output's name would be overwritten.
    """
    supervised = [o for o in outputs if not isinstance(o, UnsupervisedModelOutput)]
    targets = {
        output.target_property: batch[output.target_property]
        for output in supervised
        if output.target_property not in skip
    }
    for output in supervised:
        for mask in output.masks:
            for key in mask.required_keys:
                if key not in batch:
                    raise KeyError(
                        f"{type(mask).__name__} of output {output.name!r} reads "
                        f"{key!r}, which the batch lacks"
                    )
                targets[key] = batch[key]
        for key in output.loss_keys:  # e.g. a loss weight; taken if the batch has it
            if key in batch:
                targets[key] = batch[key]
    return targets


def predict_without_postprocessing(
    model: "AtomisticModel", batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Run ``model`` on ``batch`` with its postprocessing switched off.

    The loss compares the model's raw outputs with the targets, so the
    postprocessors (e.g. adding back the mean energy) must not run. The flag is
    restored afterwards, even if the forward pass fails. Call
    :func:`extract_targets` first, as the forward pass writes into ``batch``.
    """
    pp = model.do_postprocessing
    model.do_postprocessing = False
    try:
        return model(batch)
    finally:
        model.do_postprocessing = pp


def calculate_loss(
    outputs: list[ModelOutput],
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Weighted composite loss over all outputs."""
    loss: torch.Tensor | float = 0.0
    for output in outputs:
        loss = loss + output.calculate_loss(pred, targets)
    return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss)


def padding_mask(value: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The padding mask of a padded batch (1.0 real, 0.0 padding) that matches the
    rows of value: per atom or per structure."""
    atom_mask = batch[properties.atom_mask]
    if value.shape[0] == atom_mask.shape[0]:
        return atom_mask
    return batch[properties.structure_mask]


def calculate_padded_loss(
    outputs: list[ModelOutput],
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Weighted composite loss of a padded batch over its real rows, with fixed shapes
    and no host sync (so it can be recorded in a CUDA graph).
    """
    loss: torch.Tensor | float = 0.0
    for output in outputs:
        mask = padding_mask(pred[output.name], batch)
        loss = loss + output.calculate_padded_loss(pred, targets, mask)
    return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss)

"""
A generated variable as a model output: what a training task needs to fit it.

:class:`~schnetpack.generative.transforms.Diffuse` writes every variable's label and
loss weight into the batch; :class:`VariableOutput` compares the variable's head with
that label, so :class:`~schnetpack.lightning.AtomisticTask` (or a hand-written loop with
:func:`~schnetpack.objectives.calculate_loss`) trains a generative model like any other
output.
"""

import torch
from torchmetrics import Metric

from schnetpack import properties
from schnetpack.generative.variables import Variable
from schnetpack.objectives import ModelOutput

__all__ = ["VariableOutput"]


class VariableOutput(ModelOutput):
    """
    The head of a :class:`~schnetpack.generative.variables.Variable` as a
    :class:`~schnetpack.objectives.ModelOutput`.

    The loss is the weighted squared error per entry, averaged over rows and
    components::

        loss = sum_r w_r |head_r - label_r|^2 / (n_rows * n_components)

    with the variable's weight per row (``weight_<name>``). On a padded batch the
    weight is 0 on padding rows (:class:`~schnetpack.generative.transforms.Diffuse`
    multiplies it by the padding mask) and ``n_rows`` counts the real rows. This is
    :func:`~schnetpack.generative.variables.weighted_squared_error` divided by the
    number of components, e.g. GPFF's pseudo-force loss (mean over atoms and xyz).
    """

    def __init__(
        self,
        variable: Variable,
        loss_weight: float = 1.0,
        metrics: dict[str, Metric] | None = None,
    ):
        """
        Args:
            variable: the variable whose head is trained
            loss_weight: weight in the composite loss
            metrics: metrics of head against label (unweighted), e.g. MAE
        """
        super().__init__(
            name=variable.output_key,
            loss_weight=loss_weight,
            metrics=metrics,
            target_property=variable.label_key,
        )
        self.weight_key = variable.weight_key
        self.mask_key = (
            properties.atom_mask
            if variable.per == "atom"
            else properties.structure_mask
        )

    @property
    def loss_keys(self) -> tuple[str, ...]:
        return self.weight_key, self.mask_key

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0:
            return 0.0
        if self.weight_key not in target:
            raise KeyError(
                f"{self.weight_key!r} is missing: noise the batch with Diffuse first"
            )
        err = (pred[self.name] - target[self.target_property]) ** 2
        err = err.reshape(err.shape[0], -1)
        weight = target[self.weight_key].to(err.dtype)
        if self.mask_key in target:
            rows = target[self.mask_key].sum()
        else:
            rows = torch.tensor(float(err.shape[0]), device=err.device)
        return (
            self.loss_weight
            * (weight[:, None] * err).sum()
            / (rows.to(err.dtype) * err.shape[1])
        )

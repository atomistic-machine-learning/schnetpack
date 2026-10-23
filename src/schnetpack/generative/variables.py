"""
Generated variables: positions, and later atom types and a cell, each with
its own noising, head and loss, all sharing one time.

- :class:`Variable`: one generated quantity, shared by training and
  sampling: where it lives in the structure (its
  :class:`~schnetpack.generative.views.View`, which also moves what has to
  follow it), how it is noised, what its head predicts and how its loss is
  weighted.
- :func:`weighted_squared_error`: the default loss of a variable.

:class:`~schnetpack.generative.transforms.Diffuse` noises a list of them in
the data pipeline; :class:`~schnetpack.dynamics.sample.sampler.Sampler` steps
them with one update each. Theory: ``docs_new/generation/new_variables.md``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from schnetpack import properties
from schnetpack.generative.noising import ForwardNoising
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.views import KeyView, View

__all__ = ["Variable", "weighted_squared_error", "STRUCTURE_TIME_KEY"]

STRUCTURE_TIME_KEY = "t_structure"
"""Batch key of the time once per structure."""


def weighted_squared_error(
    prediction: torch.Tensor, label: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    The default loss of a variable: the squared error summed over the
    components of a row, weighted per row and averaged over rows.

    Args:
        prediction: the head's output, shape (n_rows, ...)
        label: the training target, shaped like prediction
        weight: loss weight per row, shape (n_rows,)
    """
    squared = (prediction - label) ** 2
    per_row = squared.reshape(squared.shape[0], -1).sum(1)
    return (weight * per_row).mean()


@dataclass(eq=False)  # compared and hashed by identity
class Variable:
    """
    One generated quantity of a structure.

    The batch holds only the structure; the variable's state is what its
    :attr:`view` reads from it. Training writes its label under
    :attr:`label_key` and its loss weight under :attr:`weight_key`; sampling
    reads its fields from the model's outputs by the keys its update is
    given.
    """

    name: str
    """Names the label, the weight and the variable in error messages."""

    view: View | str
    """Reads the state from the structure and writes it back. A batch key,
    e.g. ``properties.R``, stands for its
    :class:`~schnetpack.generative.views.KeyView`."""

    per: Literal["atom", "structure"]
    """One row per atom or one row per structure."""

    noising: ForwardNoising
    """How the variable is noised; its schedule and prior."""

    parametrization: Parametrization
    """What the head predicts: the label and the conversions."""

    output_key: str
    """Model output holding the raw head."""

    weight: Callable[..., torch.Tensor] | None = None
    """``(schedule, t, x_t) -> weight per row``; None weighs every row 1."""

    def __post_init__(self):
        if isinstance(self.view, str):
            self.view = KeyView(self.view)
        if self.per not in ("atom", "structure"):
            raise ValueError(
                f"{self.name}: per must be 'atom' or 'structure', got {self.per!r}"
            )
        schedule = getattr(self.noising, "schedule", None)
        if schedule is not None:
            self.parametrization.validate(schedule)
        prior = getattr(self.noising, "prior", None)
        per = getattr(prior, "per", self.per)
        if per != self.per:
            raise ValueError(
                f"{self.name}: the prior draws one row per {per}, the variable "
                f"has one per {self.per}; give the prior per={self.per!r}"
            )
        if self.per == "structure" and getattr(prior, "centered", False):
            raise ValueError(
                f"{self.name}: a centered prior subtracts the mean of a "
                "structure's rows, which leaves one row per structure without "
                "noise; give the prior centered=False"
            )

    @property
    def time_key(self) -> str:
        """Batch key of the time with one entry per row of this variable."""
        return properties.t if self.per == "atom" else STRUCTURE_TIME_KEY

    @property
    def label_key(self) -> str:
        return f"label_{self.name}"

    @property
    def weight_key(self) -> str:
        return f"weight_{self.name}"

    @property
    def clean_key(self) -> str:
        """Where :class:`~schnetpack.generative.transforms.Diffuse` keeps x0 if asked."""
        return f"clean_{self.name}"

    def loss(
        self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]
    ) -> torch.Tensor:
        """
        The variable's training loss from the model's outputs.

        A parametrization with a ``loss`` of its own uses it; every other
        one the :func:`weighted_squared_error`.
        """
        loss = getattr(self.parametrization, "loss", weighted_squared_error)
        return loss(
            outputs[self.output_key], batch[self.label_key], batch[self.weight_key]
        )

    def read(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """The variable's state in ``batch``."""
        return self.view.read(batch)

    def write(self, batch: Mapping[str, Any], x: torch.Tensor) -> dict[str, Any]:
        """A copy of ``batch`` with ``x`` as the variable's state."""
        return self.view.write(batch, x)

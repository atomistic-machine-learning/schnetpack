"""
Views: where a generated variable lives in the structure.

The batch holds only the structure (``Z``, ``R``, ``cell``, ...). A
variable's :class:`View` reads its state x from the batch and writes a new x
back into a copy, together with everything that has to follow it, so
``read(write(batch, x))`` is x.

- :class:`KeyView`: the state is one batch entry, e.g. ``Z``, or ``R`` of a
  molecule.
- :class:`PeriodicPositions`: the positions of a crystal, written wrapped
  into the cell.
- :class:`CellParameters`: the cell as six standardized parameters; the
  atoms follow it at fixed fractional coordinates.

:class:`ViewInput` writes a view's read into a network's inputs, for a
network conditioned on it.
"""

import abc
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from schnetpack import properties
from schnetpack.utils.cell import (
    atom_cells,
    cell_to_params,
    params_to_cell,
    valid_params,
    with_cell,
    wrap,
)

__all__ = ["View", "KeyView", "PeriodicPositions", "CellParameters", "ViewInput"]


class View(abc.ABC):
    """Reads a variable's state from a structure and writes it back."""

    @abc.abstractmethod
    def read(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """The state, one row per atom or per structure."""
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, batch: Mapping[str, Any], x: torch.Tensor) -> dict[str, Any]:
        """A copy of ``batch`` with the state ``x`` and what follows it."""
        raise NotImplementedError


class KeyView(View):
    """The state is the batch entry under ``key``; nothing follows it."""

    def __init__(self, key: str):
        """
        Args:
            key: batch key of the state, e.g. ``properties.R``
        """
        self.key = key

    def read(self, batch):
        return batch[self.key]

    def write(self, batch, x):
        return {**batch, self.key: x}

    def __repr__(self):
        return f"{type(self).__name__}({self.key!r})"


class PeriodicPositions(View):
    """
    The positions of a crystal, periodic in all three directions: written
    wrapped into each structure's cell, so ``read(write(batch, x))`` is x for
    positions inside the cell.
    """

    def read(self, batch):
        return batch[properties.R]

    def write(self, batch, x):
        return {**batch, properties.R: wrap(x, atom_cells(batch))}


class CellParameters(View):
    """
    The cell as six standardized parameters,

        y = (l - mu(N)) / s,    l = (a, b, c, alpha, beta, gamma).

    mu(N) holds the mean length times N^(1/3) for a, b and c, and the mean
    angle for alpha, beta and gamma; s^2 holds the length and angle variances
    divided by ``shrink``. A unit Gaussian prior on y is the prior
    N(mu(N), diag s^2) on the parameters in their own units (Angstrom,
    degrees), implemented as a standardization so that the interpolant stays
    zero-mean and unit-scale. The prior fixes the volume per atom rather than
    the cell's size. All six parameters are free, as in space group P1.
    Theory: ``docs_new/generation/new_variables.md`` §5.1.

    :meth:`write` builds the cell of y in the orientation and handedness of
    the cell it replaces, and the atoms keep their fractional coordinates, so
    ``write(batch, read(batch))`` is ``batch``. A structure without a cell
    yet (a zero cell, as
    :class:`~schnetpack.dynamics.structures.StatisticsStructures` gives it)
    gets the standard orientation of
    :func:`~schnetpack.utils.cell.params_to_cell`.
    """

    def __init__(
        self,
        length: float,
        angle: float,
        var_length: float,
        var_angle: float,
        shrink: float = 64.0,
    ):
        """
        Args:
            length: mean cell length divided by N^(1/3), in Angstrom
            angle: mean angle, in degrees
            var_length: variance of the length divided by N^(1/3), in A^2
            var_angle: variance of the angles, in deg^2
            shrink: the variances are divided by it, which narrows the prior

        :func:`~schnetpack.utils.cell.cell_statistics` gives the first four
        for a set of cells.
        """
        self.length = length
        self.angle = angle
        self.var_length = var_length
        self.var_angle = var_angle
        self.shrink = shrink

    def mean(self, n_atoms: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """mu(N) per structure, (n_structures, 6), in the dtype of ``like``."""
        n = n_atoms.to(like).reshape(-1, 1)
        lengths = self.length * n.pow(1.0 / 3.0).expand(-1, 3)
        return torch.cat([lengths, torch.full_like(lengths, self.angle)], -1)

    def std(self, like: torch.Tensor) -> torch.Tensor:
        """s of the six parameters, (6,), in the dtype of ``like``."""
        std_length = math.sqrt(self.var_length / self.shrink)
        std_angle = math.sqrt(self.var_angle / self.shrink)
        return torch.tensor(
            [std_length] * 3 + [std_angle] * 3, dtype=like.dtype, device=like.device
        )

    def params(self, y: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
        """Cell parameters from y, (n_structures, 6)."""
        return self.mean(n_atoms, y) + self.std(y) * y

    def read(self, batch):
        """y of every structure, (n_structures, 6)."""
        params = cell_to_params(batch[properties.cell].view(-1, 3, 3))
        mean = self.mean(batch[properties.n_atoms], params)
        return (params - mean) / self.std(params)

    def write(self, batch, y):
        """
        The cells of y, in the orientation of the old ones, with the atoms at
        their fractional coordinates.

        Raises:
            ValueError: if y gives parameters no cell has.
        """
        if properties.cell in batch:
            y = y.to(batch[properties.cell])  # the structure's dtype and device
        params = self.params(y, batch[properties.n_atoms])
        valid = valid_params(params)
        if not valid.all():
            raise ValueError(
                f"{int((~valid).sum())} of {valid.numel()} structures have "
                f"cell parameters no cell has, e.g. "
                f"{[round(p, 3) for p in params[~valid][0].tolist()]} "
                "(a, b, c, alpha, beta, gamma)"
            )
        cell = params_to_cell(params)
        if properties.cell not in batch:
            return with_cell(batch, cell)
        old = batch[properties.cell].view(-1, 3, 3)
        placed = old.abs().sum((-2, -1)) > 0
        # old = standard @ Q with Q orthogonal: both have the same metric
        rotation = torch.eye(3, dtype=cell.dtype, device=cell.device).repeat(
            cell.shape[0], 1, 1
        )
        if placed.any():
            standard = params_to_cell(cell_to_params(old[placed]))
            rotation[placed] = torch.linalg.solve(standard, old[placed])
        return with_cell(batch, cell @ rotation)


class ViewInput(nn.Module):
    """
    Writes a view's read of the inputs under ``key``: an input module for a
    network conditioned on a variable its representation does not see, such
    as the cell parameters of a crystal larger than the cutoff.
    """

    def __init__(self, view: View, key: str):
        """
        Args:
            view: the view to read
            key: input key the state is written to
        """
        super().__init__()
        self.view = view
        self.key = key

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs[self.key] = self.view.read(inputs)
        return inputs

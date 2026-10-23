"""
Periodic geometry: cell parameters, fractional coordinates, wrapping and the
Niggli cell.

The lattice vectors are the rows of a cell, so a position is ``R = frac @ cell``.
Functions on positions take one cell per position, (n, 3, 3), or one cell for
all, (3, 3); :func:`atom_cells` gives the first for a structure or a batch.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from ase.cell import Cell

from schnetpack import properties

__all__ = [
    "cell_to_params",
    "params_to_cell",
    "valid_params",
    "cell_statistics",
    "niggli_reduce",
    "structure_index",
    "atom_cells",
    "with_cell",
    "to_frac",
    "to_cart",
    "wrap",
    "minimum_image",
]


def cell_to_params(cell: torch.Tensor) -> torch.Tensor:
    """
    The six parameters of cells.

    Args:
        cell: cells, shape (..., 3, 3), rows are lattice vectors

    Returns:
        (..., 6): lengths a, b, c in the cell's unit, then the angles alpha
        (between b and c), beta (a, c) and gamma (a, b) in degrees.
    """
    a, b, c = cell.unbind(-2)
    lengths = torch.stack([a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)], -1)

    def angle(u, v):
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        # inside (-1, 1): arccos has an infinite slope at the ends
        return torch.rad2deg(torch.arccos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)))

    angles = torch.stack([angle(b, c), angle(a, c), angle(a, b)], -1)
    return torch.cat([lengths, angles], -1)


def params_to_cell(params: torch.Tensor) -> torch.Tensor:
    """
    Cells in standard orientation from their parameters: a along x, b in the
    xy-plane, c with a positive z component.

    Args:
        params: (..., 6) as returned by :func:`cell_to_params`

    Returns:
        (..., 3, 3), rows are lattice vectors; right-handed.
    """
    a, b, c = params[..., :3].unbind(-1)
    alpha, beta, gamma = torch.deg2rad(params[..., 3:]).unbind(-1)
    cos_al, cos_be, cos_ga = alpha.cos(), beta.cos(), gamma.cos()
    sin_ga = gamma.sin()
    cy = (cos_al - cos_be * cos_ga) / sin_ga
    # zero for angles no cell can have, which the clamp keeps finite
    cz = torch.sqrt((1.0 - cos_be**2 - cy**2).clamp(min=1e-12))
    zero = torch.zeros_like(a)
    return torch.stack(
        [
            torch.stack([a, zero, zero], -1),
            torch.stack([b * cos_ga, b * sin_ga, zero], -1),
            torch.stack([c * cos_be, c * cy, c * cz], -1),
        ],
        -2,
    )


def valid_params(params: torch.Tensor) -> torch.Tensor:
    """
    Whether parameters describe a cell: positive lengths, angles in (0, 180)
    degrees and a positive volume. :func:`params_to_cell` builds a flat cell
    for any other.

    Args:
        params: (..., 6) as returned by :func:`cell_to_params`

    Returns:
        (...,) bool
    """
    lengths, angles = params[..., :3], params[..., 3:]
    cos_al, cos_be, cos_ga = torch.deg2rad(angles).cos().unbind(-1)
    # (V / abc)^2
    volume = 1.0 - cos_al**2 - cos_be**2 - cos_ga**2 + 2.0 * cos_al * cos_be * cos_ga
    return (
        (lengths > 0).all(-1)
        & (angles > 0).all(-1)
        & (angles < 180).all(-1)
        & (volume > 0)
    )


def cell_statistics(cells: torch.Tensor, n_atoms: torch.Tensor) -> dict[str, float]:
    """
    Mean and variance of the cell lengths divided by N^(1/3), and of the
    angles, pooled over the three lengths and the three angles of every
    cell: the statistics of
    :class:`~schnetpack.generative.views.CellParameters`. Take them on the
    cells as training sees them, e.g. Niggli-reduced.

    Args:
        cells: (n_structures, 3, 3), rows are lattice vectors
        n_atoms: atoms per structure, (n_structures,)

    Returns:
        ``length`` and ``var_length`` in the cells' unit, ``angle`` and
        ``var_angle`` in degrees.
    """
    params = cell_to_params(cells.reshape(-1, 3, 3).double())
    n = n_atoms.double().reshape(-1, 1)
    lengths, angles = params[:, :3] / n.pow(1.0 / 3.0), params[:, 3:]
    return {
        "length": float(lengths.mean()),
        "angle": float(angles.mean()),
        "var_length": float(lengths.var()),
        "var_angle": float(angles.var()),
    }


def niggli_reduce(cell: torch.Tensor) -> torch.Tensor:
    """
    The Niggli-reduced basis of one lattice, in the cell's own orientation.

    Args:
        cell: (3, 3), rows are lattice vectors

    Returns:
        (3, 3): integer combinations of the rows, with determinant +1, that
        form the lattice's Niggli cell. Fractional coordinates taken against
        it describe the same crystal.
    """
    _, op = Cell(cell.detach().cpu().numpy()).niggli_reduce()
    # ase returns the reduced cell rotated to its own standard orientation;
    # op^T @ cell is the same basis in the input's orientation
    op = torch.as_tensor(np.ascontiguousarray(op.T), dtype=cell.dtype)
    return op.to(cell.device) @ cell


def structure_index(inputs: Mapping[str, Any]) -> torch.Tensor:
    """
    The structure of every atom: ``idx_m`` in a batch; zeros in a single
    structure, as a transform sees it.
    """
    if properties.idx_m in inputs:
        return inputs[properties.idx_m]
    if properties.Z in inputs:
        return torch.zeros_like(inputs[properties.Z])
    r = inputs[properties.R]
    return torch.zeros(r.shape[0], dtype=torch.long, device=r.device)


def atom_cells(inputs: Mapping[str, Any]) -> torch.Tensor:
    """The cell of every atom's structure, shape (n_atoms, 3, 3)."""
    return inputs[properties.cell].view(-1, 3, 3)[structure_index(inputs)]


def with_cell(inputs: Mapping[str, Any], cell: torch.Tensor) -> dict[str, Any]:
    """
    A copy of ``inputs`` with a new cell, every atom at its old fractional
    coordinates, so the atoms move with the cell. Without positions only the
    cell is replaced.

    Args:
        inputs: a structure or a batch; its positions need a regular cell
        cell: the new cells, (n_structures, 3, 3)
    """
    inputs = dict(inputs)
    if properties.R in inputs:
        frac = to_frac(inputs[properties.R], atom_cells(inputs))
        inputs[properties.R] = to_cart(
            frac, cell.view(-1, 3, 3)[structure_index(inputs)]
        )
    inputs[properties.cell] = cell
    return inputs


def to_frac(r: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """
    Fractional coordinates, R = frac @ cell.

    Args:
        r: positions, (n, 3)
        cells: one cell per position, (n, 3, 3), or one for all, (3, 3)
    """
    return (r.unsqueeze(-2) @ torch.linalg.inv(cells)).squeeze(-2)


def to_cart(frac: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Positions from fractional coordinates; ``cells`` as in :func:`to_frac`."""
    return (frac.unsqueeze(-2) @ cells).squeeze(-2)


def wrap(r: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Positions moved into their cell, fractional coordinates in [0, 1)."""
    return to_cart(to_frac(r, cells) % 1.0, cells)


def minimum_image(d: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """
    Displacements shifted by whole lattice vectors to fractional components
    in [-1/2, 1/2]: the shortest image for a reduced cell, a short one for
    any other.
    """
    frac = to_frac(d, cells)
    return to_cart(frac - torch.round(frac), cells)

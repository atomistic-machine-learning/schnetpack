"""
The cell as three more rows of an optimizer's variables.

The variables of ase's ``FrechetCellFilter``, for a batch. A structure of n
atoms has n + 3 rows: the positions in the cell the run started from, and
n log F, where the deformation F takes that cell to the current one. Positions
and cell follow from the rows as r = F x and cell = cell0 F^T. The rows of a
batch are the atom rows of every structure end to end, then three cell rows per
structure.

The factor n puts the cell rows on the scale of the atom rows, so a step rule
and a stop test can treat them as three more atoms.
"""

import torch

__all__ = ["pack", "unpack", "generalized_forces"]


def _deformation(cell_rows: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """F = exp(rows / n) per structure, ``(n_structures, 3, 3)``."""
    return torch.linalg.matrix_exp(cell_rows.view(-1, 3, 3) / n_atoms[:, None, None])


def pack(
    positions: torch.Tensor,
    cell_rows: torch.Tensor,
    idx_m: torch.Tensor,
    n_atoms: torch.Tensor,
) -> torch.Tensor:
    """
    The rows of a batch: its positions taken back to the starting cell, then
    the cell rows.

    Args:
        positions: ``(n_total_atoms, 3)`` current positions
        cell_rows: ``(3 * n_structures, 3)`` n log F of every structure
        idx_m: ``(n_total_atoms,)`` structure of every atom
        n_atoms: ``(n_structures,)`` atoms per structure

    Returns:
        ``(n_total_atoms + 3 * n_structures, 3)`` rows, in ``cell_rows``' dtype.
    """
    deformation = _deformation(cell_rows, n_atoms)
    reference = torch.linalg.solve(
        deformation[idx_m], positions.to(cell_rows.dtype).unsqueeze(-1)
    )
    return torch.cat([reference.squeeze(-1), cell_rows])


def unpack(
    x: torch.Tensor, cell0: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The positions and cells the rows ``x`` stand for.

    Args:
        x: rows as :func:`pack` returns them
        cell0: ``(n_structures, 3, 3)`` cells the run started from
        idx_m: ``(n_total_atoms,)`` structure of every atom
        n_atoms: ``(n_structures,)`` atoms per structure

    Returns:
        ``(n_total_atoms, 3)`` positions and ``(n_structures, 3, 3)`` cells.
    """
    n_total = idx_m.shape[0]
    deformation = _deformation(x[n_total:], n_atoms)
    positions = (deformation[idx_m] @ x[:n_total].unsqueeze(-1)).squeeze(-1)
    return positions, cell0 @ deformation.mT


def generalized_forces(
    forces: torch.Tensor,
    stress: torch.Tensor,
    cell: torch.Tensor,
    cell_rows: torch.Tensor,
    idx_m: torch.Tensor,
    n_atoms: torch.Tensor,
) -> torch.Tensor:
    """
    Minus the gradient of the energy by the rows.

    Args:
        forces: ``(n_total_atoms, 3)`` forces on the atoms
        stress: ``(n_structures, 3, 3)`` stress, positive under tension
        cell: ``(n_structures, 3, 3)`` current cells
        cell_rows: ``(3 * n_structures, 3)`` n log F of every structure
        idx_m: ``(n_total_atoms,)`` structure of every atom
        n_atoms: ``(n_structures,)`` atoms per structure

    Returns:
        Rows shaped like :func:`pack`'s: F^T f for every atom, then the force
        on log F over n.
    """
    dtype = cell_rows.dtype
    virial = -torch.linalg.det(cell).abs()[:, None, None] * stress
    with torch.enable_grad():
        log_deformation = (
            cell_rows.view(-1, 3, 3) / n_atoms[:, None, None]
        ).requires_grad_()
        deformation = torch.linalg.matrix_exp(log_deformation)
        # the virial is the force on a strain of the current cell, which makes
        # virial F^-T the force on F; the chain rule through the matrix
        # exponential takes it on to log F
        on_deformation = virial.to(dtype) @ torch.linalg.inv(deformation.detach()).mT
        (on_log,) = torch.autograd.grad(
            (deformation * on_deformation).sum(), log_deformation
        )
    deformation = deformation.detach()
    on_atoms = (forces.to(dtype).unsqueeze(1) @ deformation[idx_m]).squeeze(1)
    return torch.cat([on_atoms, (on_log / n_atoms[:, None, None]).view(-1, 3)])

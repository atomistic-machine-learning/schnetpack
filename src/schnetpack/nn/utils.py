from typing import Callable, Optional

import torch
from torch import nn as nn

__all__ = ["replicate_module", "derivative_from_atomic", "derivative_from_molecular"]

from torch.autograd import grad


def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list


def derivative_from_molecular(
    fx: torch.Tensor,
    dx: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = False,
):
    """
    Compute the derivative of `fx` with respect to `dx` if the leading dimension of `fx` is the number of molecules
    (e.g. energies, dipole moments, etc).

    Args:
        fx (torch.Tensor): Tensor for which the derivative is taken.
        dx (torch.Tensor): Derivative.
        create_graph (bool): Create computational graph.
        retain_graph (bool): Keep the computational graph.

    Returns:
        torch.Tensor: derivative of `fx` with respect to `dx`.
    """
    fx_shape = fx.shape
    dx_shape = dx.shape
    # Final shape takes into consideration whether derivative will yield atomic or molecular properties
    final_shape = (dx_shape[0], *fx_shape[1:], *dx_shape[1:])

    fx = fx.view(fx_shape[0], -1)

    dfdx = torch.stack(
        [
            grad(
                fx[..., i],
                dx,
                torch.ones_like(fx[..., i]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]
            for i in range(fx.shape[1])
        ],
        dim=1,
    )
    dfdx = dfdx.view(final_shape)

    return dfdx


def derivative_from_atomic(
    fx: torch.Tensor,
    dx: torch.Tensor,
    n_atoms: torch.Tensor,
    reduce_ij: bool,
    create_graph: bool = False,
    retain_graph: bool = False,
    idx_i: Optional[torch.Tensor] = None,
    idx_j: Optional[torch.Tensor] = None,
):
    """
    Compute the derivative of a tensor with the leading dimension of (batch x atoms) with respect to another tensor of
    either dimension (batch * atoms) (e.g. R) or (batch * atom pairs) (e.g. Rij). This function is primarily used for
    computing Hessians and Hessian-like response properties (e.g. nuclear spin-spin couplings). The final tensor will
    have the shape ( batch * atoms * atoms x ....).

    This is quite inefficient, use with care.

    Args:
        fx (torch.Tensor): Tensor for which the derivative is taken.
        dx (torch.Tensor): Derivative.
        n_atoms (torch.Tensor): Tensor containing the number of atoms for each molecule.
        reduce_ij (bool): If the derivative is with respect to distances R_ij, reduce to atom positions.
        create_graph (bool): Create computational graph.
        retain_graph (bool): Keep the computational graph.
        idx_i (torch.Tensor, optional): Index for i neighbors, required for `reduce_ij`.
        idx_j (torch.Tensor, optional): Index for j neighbors, required for `reduce_ij`.

    Returns:
        torch.Tensor: derivative of `fx` with respect to `dx`.
    """
    n_total = torch.sum(n_atoms)

    # Split input tensor for easier bookkeeping
    fxm = fx.split(list(n_atoms))

    # create_graph = True

    dfdx = []

    n_mol = 0
    # Compute all derivatives
    for idx in range(len(fxm)):
        fx = fxm[idx].view(-1)

        # Generate the individual derivatives
        dfdx_mol = []
        for i in range(fx.shape[0]):
            dfdx_i = grad(
                fx[i],
                dx,
                torch.ones_like(fx[i]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            # Reduce Rij to R if requested.
            if reduce_ij:
                dfdx_i_tmp = torch.zeros(
                    n_total, *dfdx_i.shape[1:], device=fx.device, dtype=fx.dtype
                )
                dfdx_i_tmp = dfdx_i_tmp.index_add(0, idx_i, -dfdx_i)
                dfdx_i = dfdx_i_tmp.index_add(0, idx_j, dfdx_i)

            dfdx_mol.append(dfdx_i[n_mol : n_mol + n_atoms[idx], ...])

        # Build molecular matrix and reshape
        dfdx_mol = torch.stack(dfdx_mol, dim=0)
        dfdx_mol = dfdx_mol.view(n_atoms[idx], 3, n_atoms[idx], 3)
        dfdx_mol = dfdx_mol.permute(0, 2, 1, 3)
        dfdx_mol = dfdx_mol.reshape(n_atoms[idx] ** 2, 3, 3)

        dfdx.append(dfdx_mol)

        n_mol += n_atoms[idx]

    # Accumulate everything
    dfdx = torch.cat(dfdx, dim=0)

    return dfdx

import math
import torch
from sympy.physics.wigner import clebsch_gordan

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=10)
def sh_indices(lmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build index arrays for spherical harmonics

    Args:
        lmax: maximum angular momentum
    """
    ls = torch.arange(0, lmax + 1)
    nls = 2 * ls + 1
    lidx = torch.repeat_interleave(ls, nls)
    midx = torch.cat([torch.arange(-l, l + 1) for l in ls])
    return lidx, midx


@lru_cache(maxsize=10)
def generate_sh_to_rsh(lmax: int) -> torch.Tensor:
    """
    Generate transformation matrix to convert (complex) spherical harmonics to real form

    Args:
        lmax: maximum angular momentum
    """
    lidx, midx = sh_indices(lmax)
    l1 = lidx[:, None]
    l2 = lidx[None, :]
    m1 = midx[:, None]
    m2 = midx[None, :]
    U = (
        1.0 * ((m1 == 0) * (m2 == 0))
        + (-1.0) ** abs(m1) / math.sqrt(2) * ((m1 == m2) * (m1 > 0))
        + 1.0 / math.sqrt(2) * ((m1 == -m2) * (m2 < 0))
        + -1.0j * (-1.0) ** abs(m1) / math.sqrt(2) * ((m1 == -m2) * (m1 < 0))
        + 1.0j / math.sqrt(2) * ((m1 == m2) * (m1 < 0))
    ) * (l1 == l2)
    return U


@lru_cache(maxsize=10)
def generate_clebsch_gordan(lmax: int) -> torch.Tensor:
    """
    Generate standard Clebsch-Gordan coefficients for complex spherical harmonics

    Args:
        lmax: maximum angular momentum
    """
    lidx, midx = sh_indices(lmax)
    cg = torch.zeros((lidx.shape[0], lidx.shape[0], lidx.shape[0]))
    lidx = lidx.numpy()
    midx = midx.numpy()
    for c1, (l1, m1) in enumerate(zip(lidx, midx)):
        for c2, (l2, m2) in enumerate(zip(lidx, midx)):
            for c3, (l3, m3) in enumerate(zip(lidx, midx)):
                if abs(l1 - l2) <= l3 <= min(l1 + l2, lmax) and m3 in {
                    m1 + m2,
                    m1 - m2,
                    m2 - m1,
                    -m1 - m2,
                }:
                    coeff = clebsch_gordan(l1, l2, l3, m1, m2, m3)
                    cg[c1, c2, c3] = float(coeff)
    return cg


@lru_cache(maxsize=10)
def generate_clebsch_gordan_rsh(
    lmax: int, parity_invariance: bool = True
) -> torch.Tensor:
    """
    Generate Clebsch-Gordan coefficients for real spherical harmonics

    Args:
        lmax: maximum angular momentum
        parity_invariance: whether to enforce parity invariance, i.e. only allow
            non-zero coefficients if :math:`-1^l_1 -1^l_2 = -1^l_3`

    """
    lidx, _ = sh_indices(lmax)
    cg = generate_clebsch_gordan(lmax).to(dtype=torch.complex64)
    complex_to_real = generate_sh_to_rsh(lmax)  # (real, complex)
    cg_rsh = torch.einsum(
        "ijk,mi,nj,ok->mno",
        cg,
        complex_to_real,
        complex_to_real,
        complex_to_real.conj(),
    )

    if parity_invariance:
        parity = (-1.0) ** lidx
        pmask = parity[:, None, None] * parity[None, :, None] == parity[None, None, :]
        cg_rsh *= pmask
    else:
        lsum = lidx[:, None, None] + lidx[None, :, None] - lidx[None, None, :]
        cg_rsh *= 1.0j**lsum

    # cast to real
    cg_rsh = cg_rsh.real.to(torch.float64)
    return cg_rsh


def sparsify_clebsch_gordon(
    cg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Clebsch-Gordon tensor to sparse format.

    Args:
        cg: dense tensor Clebsch-Gordon coefficients
            [(lmax_1+1)^2, (lmax_2+1)^2, (lmax_out+1)^2]

    Returns:
        cg_sparse: vector of non-zeros CG coefficients
        idx_in_1: indices for first set of irreps
        idx_in_2: indices for second set of irreps
        idx_out: indices for output set of irreps
    """
    idx = torch.nonzero(cg)
    idx_in_1, idx_in_2, idx_out = torch.split(idx, 1, dim=1)
    idx_in_1, idx_in_2, idx_out = (
        idx_in_1[:, 0],
        idx_in_2[:, 0],
        idx_out[:, 0],
    )
    cg_sparse = cg[idx_in_1, idx_in_2, idx_out]
    return cg_sparse, idx_in_1, idx_in_2, idx_out


def round_cmp(x: torch.Tensor, decimals: int = 1):
    return torch.round(x.real, decimals=decimals) + 1j * torch.round(
        x.imag, decimals=decimals
    )

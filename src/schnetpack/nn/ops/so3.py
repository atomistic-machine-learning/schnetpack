import math
import numpy as np
import torch
from sympy.physics.wigner import clebsch_gordan

from functools import lru_cache


@lru_cache(maxsize=10)
def sh_indices(lmax: int):
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
def generate_sh_to_rsh(lmax: int) -> np.ndarray:
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
def generate_clebsch_gordan(lmax: int) -> np.ndarray:
    """
    Generate standard Clebsch-Gordan coefficients for complex spherical harmonics

    Args:
        lmax: maximum angular momentum
    """
    lidx, midx = sh_indices(lmax)

    cg = torch.zeros((lidx.shape[0], lidx.shape[0], lidx.shape[0]))
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
def generate_clebsch_gordan_rsh(lmax: int, parity_mode: str = "mask") -> np.ndarray:
    """
    Generate Clebsch-Gordan coefficients for real spherical harmonics

    Args:
        lmax: maximum angular momentum
        parity_mode: treatment of odd parity:
            * 'mask': set to zero
            * 'realize': convert imaginary values to real

    """
    cg = generate_clebsch_gordan(lmax).to(dtype=torch.complex64)
    U = generate_sh_to_rsh(lmax)
    cg_rsh = torch.einsum("ijk,mi,nj,ok->mno", cg, U, U, U.conj())

    lidx, _ = sh_indices(lmax)
    if parity_mode == "mask":
        parity = (-1.0) ** lidx
        pmask = parity[:, None, None] * parity[None, :, None] == parity[None, None, :]
        cg_rsh *= pmask
    elif parity_mode == "realize":
        lsum = lidx[:, None, None] + lidx[None, :, None] - lidx[None, None, :]
        cg_rsh *= 1.0j**lsum
    else:
        raise ValueError('Argument `parity_mode` has to be one of ["mask", "realize"]')

    # cast to real
    cg_rsh = cg_rsh.real.to(torch.float64)
    return cg_rsh

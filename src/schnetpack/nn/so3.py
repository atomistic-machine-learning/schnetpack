import math
import torch
import torch.nn as nn
import schnetpack.nn as snn
from .ops.so3 import generate_clebsch_gordan_rsh, sh_indices
from .ops.math import binom
from schnetpack.utils import as_dtype


class RealSphericalHarmonics(nn.Module):
    """
    Generates the real spherical harmonics for a batch of (normalized) vectors.

    Spherical harmonics are generated up to angular momentum `lmax` in dimension 1,
    according to the following order:
    - l=0, m=0
    - l=1, m=-1
    - l=1, m=0
    - l=1, m=1
    - l=2, m=-2
    - l=2, m=-1
    - etc.
    """

    def __init__(self, lmax: int, dtype_str: str = "float32"):
        super().__init__()
        self.lmax = lmax

        (
            powers,
            zpow,
            cAm,
            cBm,
            cPi,
        ) = self.generate_Ylm_coefficients(lmax)

        dtype = as_dtype(dtype_str)
        self.register_buffer("powers", powers.to(dtype=dtype), False)
        self.register_buffer("zpow", zpow.to(dtype=dtype), False)
        self.register_buffer("cAm", cAm.to(dtype=dtype), False)
        self.register_buffer("cBm", cBm.to(dtype=dtype), False)
        self.register_buffer("cPi", cPi.to(dtype=dtype), False)

        ls = torch.arange(0, lmax + 1)
        nls = 2 * ls + 1
        self.lidx = torch.repeat_interleave(ls, nls)
        self.midx = torch.cat([torch.arange(-l, l + 1) for l in ls])

        self.register_buffer("flidx", self.lidx.to(dtype=dtype), False)

    def generate_Ylm_coefficients(self, lmax: int):
        # see: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_forms

        # calculate Am/Bm coefficients
        m = torch.arange(1, lmax + 1, dtype=torch.float64)[:, None]
        p = torch.arange(0, lmax + 1, dtype=torch.float64)[None, :]
        mask = p <= m
        mCp = binom(m, p)
        cAm = mCp * torch.cos(0.5 * math.pi * (m - p))
        cBm = mCp * torch.sin(0.5 * math.pi * (m - p))
        cAm *= mask
        cBm *= mask
        powers = torch.stack([torch.broadcast_to(p, cAm.shape), m - p], dim=-1)
        powers *= mask[:, :, None]

        # calculate Pi coefficients
        l = torch.arange(0, lmax + 1, dtype=torch.float64)[:, None, None]
        m = torch.arange(0, lmax + 1, dtype=torch.float64)[None, :, None]
        k = torch.arange(0, lmax // 2 + 1, dtype=torch.float64)[None, None, :]
        cPi = torch.sqrt(torch.exp(torch.lgamma(l - m + 1) - torch.lgamma(l + m + 1)))
        cPi = cPi * (-1) ** k * 2 ** (-l) * binom(l, k) * binom(2 * l - 2 * k, l)
        cPi *= torch.exp(torch.lgamma(l - 2 * k + 1) - torch.lgamma(l - 2 * k - m + 1))
        zpow = l - 2 * k - m

        # masking of invalid entries
        cPi = torch.nan_to_num(cPi, 100.0)
        mask1 = k <= torch.floor((l - m) / 2)
        mask2 = l >= m
        mask = mask1 * mask2
        cPi *= mask
        zpow *= mask

        return powers, zpow, cAm, cBm, cPi

    def forward(self, R: torch.Tensor):
        target_shape = [
            R.shape[0],
            self.powers.shape[0],
            self.powers.shape[1],
            2,
        ]
        Rs = torch.broadcast_to(R[:, None, None, :2], target_shape)
        pows = torch.broadcast_to(self.powers[None], target_shape)

        Rs = torch.where(pows == 0, torch.ones_like(Rs), Rs)

        temp = Rs**self.powers
        monomials_xy = torch.prod(temp, dim=-1)

        Am = torch.sum(monomials_xy * self.cAm[None], 2)
        Bm = torch.sum(monomials_xy * self.cBm[None], 2)
        ABm = torch.cat(
            [
                torch.flip(Bm, (1,)),
                math.sqrt(0.5) * torch.ones((Am.shape[0], 1), device=R.device),
                Am,
            ],
            dim=1,
        )
        ABm = ABm[:, self.midx + self.lmax]

        target_shape = [
            R.shape[0],
            self.zpow.shape[0],
            self.zpow.shape[1],
            self.zpow.shape[2],
        ]
        z = torch.broadcast_to(R[:, 2, None, None, None], target_shape)
        zpows = torch.broadcast_to(self.zpow[None], target_shape)
        z = torch.where(zpows == 0, torch.ones_like(z), z)
        zk = z**zpows

        Pi = torch.sum(zk * self.cPi, dim=-1)  # batch x L x M
        Pi_lm = Pi[:, self.lidx, abs(self.midx)]
        sphharm = torch.sqrt((2 * self.flidx + 1) / (2 * math.pi)) * Pi_lm * ABm
        return sphharm


def scalar2rsh(x: torch.Tensor, lmax: int) -> torch.Tensor:
    """
    Expand scalar tensor to spherical harmonics shape with angular momentum up to `lmax`
    """
    y = torch.cat(
        [
            x,
            torch.zeros(
                (x.shape[0], (lmax + 1) ** 2 - 1, x.shape[2]),
                device=x.device,
                dtype=x.dtype,
            ),
        ],
        dim=1,
    )
    return y


class BaseSO3Convolution(nn.Module):
    """
    Base class for SO3-equivariant convolutions.
    """

    def __init__(self, lmax: int, n_atom_basis: int, n_radial: int):
        super().__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis
        self.n_radial = n_radial

        cg = torch.from_numpy(generate_clebsch_gordan_rsh(lmax))
        cg = cg.to(torch.float32)

        idx = torch.nonzero(cg)
        idx_in_1, idx_in_2, idx_out = torch.split(idx, 1, dim=1)
        idx_in_1, idx_in_2, idx_out = (
            idx_in_1[:, 0],
            idx_in_2[:, 0],
            idx_out[:, 0],
        )

        self.register_buffer("idx_in_1", idx_in_1, persistent=False)
        self.register_buffer("idx_in_2", idx_in_2, persistent=False)
        self.register_buffer("idx_out", idx_out, persistent=False)

        self.register_buffer(
            "clebsch_gordan",
            cg[self.idx_in_1, self.idx_in_2, self.idx_out],
            persistent=False,
        )

    def compute_radial_filter(
        self, radial_ij: torch.Tensor, cutoff_ij: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        radial_ij: torch.Tensor,
        dir_ij: torch.Tensor,
        cutoff_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
    ) -> torch.Tensor:
        xj = x[idx_j[:, None], self.idx_in_2[None, :], :]
        Wij = self.compute_radial_filter(radial_ij, cutoff_ij)

        v = (
            Wij
            * dir_ij[:, self.idx_in_1, None]
            * self.clebsch_gordan[None, :, None]
            * xj
        )
        yij = snn.scatter_add(v, self.idx_out, dim_size=(self.lmax + 1) ** 2, dim=1)
        y = snn.scatter_add(yij, idx_i, dim_size=n_atoms)
        return y


class SO3Convolution(BaseSO3Convolution):
    r"""
    SO3-equivariant convolution

    y - shape: atom, spherical harmonic, feature

    .. math::

        y_{i,s,f} = \sum_{j,s_1,s_2} x_{j,s_2,f} W_{s_1,f}(r_{ij}) Y_{s_1}(\vec{r}_{ij})
            C_{s_1,s_2}^{s}

    """

    def __init__(self, lmax: int, n_atom_basis: int, n_radial: int):
        super().__init__(lmax, n_atom_basis, n_radial)

        self.filternet = snn.Dense(
            n_radial, n_atom_basis * (self.lmax + 1), activation=None
        )

        ls = torch.arange(0, lmax + 1)
        nls = 2 * ls + 1
        lidx = torch.repeat_interleave(ls, nls)
        self.register_buffer("Widx", lidx[self.idx_in_1])

    def compute_radial_filter(
        self, radial_ij: torch.Tensor, cutoff_ij: torch.Tensor
    ) -> torch.Tensor:
        Wij = self.filternet(radial_ij) * cutoff_ij
        Wij = torch.reshape(Wij, (-1, self.lmax + 1, self.n_atom_basis))
        Wij = Wij[:, self.Widx]
        return Wij


class SO3SelfInteraction(nn.Module):
    r"""
    Self-interaction between spherical harmonics on the same atom

    y - shape: atom, spherical harmonic, feature

    .. math::

        y_{i,s,f} = \sum_{s_1,s_2} x_{j,s_1,f} x_{j,s_2,f} W_{s_1,s_2,f} C_{s_1,s_2}^{s}

    """

    def __init__(self, lmax: int, device=None, dtype=None):
        super().__init__()
        self.lmax = lmax
        cg = generate_clebsch_gordan_rsh(lmax)
        cg = cg.to(torch.float32)

        idx = torch.nonzero(cg)
        idx_1, idx_2, idx_out = torch.split(idx, 1, dim=1)
        idx_1, idx_2, idx_out = (
            idx_1[:, 0],
            idx_2[:, 0],
            idx_out[:, 0],
        )

        self.register_buffer("idx_1", idx_1, persistent=False)
        self.register_buffer("idx_2", idx_2, persistent=False)
        self.register_buffer("idx_out", idx_out, persistent=False)

        self.register_buffer(
            "clebsch_gordan",
            cg[self.idx_1, self.idx_2, self.idx_out],
            persistent=False,
        )

        ls = torch.arange(0, lmax + 1)
        nls = 2 * ls + 1
        self.lidx = torch.repeat_interleave(ls, nls)

        widx_1 = self.lidx[self.idx_1]
        widx_2 = self.lidx[self.idx_2]
        self.register_buffer("widx_1", widx_1, persistent=False)
        self.register_buffer("widx_2", widx_2, persistent=False)

        self.weight = nn.Parameter(
            torch.empty((lmax + 1, lmax + 1), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight[self.widx_1, self.widx_2]
        Wcg = W * self.clebsch_gordan

        temp = x[:, self.idx_1, :] * x[:, self.idx_2, :] * Wcg[None, :, None]
        y = snn.scatter_add(temp, self.idx_out, dim_size=(self.lmax + 1) ** 2, dim=1)
        return y


class SO3ParametricGatedNonlinearity(nn.Module):
    """
    SO3-equivariant parametric gated nonlinearity

    .. math::

        y_{i,s,f} = x_{j,s,f} * \sigma(f(x_{j,0,\cdot}))

    """

    def __init__(self, n_in: int, lmax: int):
        super().__init__()
        self.lmax = lmax
        self.n_in = n_in
        ls = torch.arange(0, lmax + 1)
        nls = 2 * ls + 1
        self.lidx = torch.repeat_interleave(ls, nls)
        self.scaling = nn.Linear(n_in, n_in * (lmax + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x[:, 0, :]
        h = self.scaling(s0).reshape(-1, self.lmax + 1, self.n_in)
        h = h[:, self.lidx]
        y = x * torch.sigmoid(h)
        return y


class SO3GatedNonlinearity(nn.Module):
    """
    SO3-equivariant gated nonlinearity

    .. math::

        y_{i,s,f} = x_{j,s,f} * \sigma(x_{j,0,\cdot})

    """

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = lmax
        ls = torch.arange(0, lmax + 1)
        nls = 2 * ls + 1
        self.lidx = torch.repeat_interleave(ls, nls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x[:, 0, :]
        y = x * torch.sigmoid(s0[:, None, :])
        return y

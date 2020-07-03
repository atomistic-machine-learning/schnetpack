import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
from schnetpack.nn.activations import softplus_inverse
import numpy as np
import os

__all__ = ["ElectrostaticEnergy", "ZBLRepulsionEnergy", "D4DispersionEnergy"]


class Correction(nn.Module):
    def __init__(self):
        super(Correction, self).__init__()

    def forward(self, inputs):
        pass


class ElectrostaticEnergy(nn.Module):
    def __init__(self, cuton, cutoff, ke=14.399645351950548, lr_cutoff=None):
        # todo: check initial ke value or set to none
        # todo: *0.5
        super(ElectrostaticEnergy, self).__init__()
        self.ke = ke
        self.cuton = cuton
        self.cutoff = cutoff
        self.lr_cutoff = lr_cutoff
        # these are constants for when a lr_cutoff is used
        if lr_cutoff is not None:
            self.cut_rconstant = lr_cutoff ** 15 / (lr_cutoff ** 16 + cuton ** 16) ** (
                17 / 16
            )
            self.cut_constant = 1 / (cuton ** 16 + lr_cutoff ** 16) ** (
                1 / 16
            ) + lr_cutoff ** 16 / (lr_cutoff ** 16 + cuton ** 16) ** (17 / 16)

    def forward(self, inputs, atomwise_predictions):
        # todo: collect neigh_elements or something
        # get properties
        qi = atomwise_predictions["qi"]
        r_ij = inputs["distances"]
        neighbors = inputs[spk.Properties.neighbors]
        neighbor_mask = inputs[spk.Properties.neighbor_mask]

        # todo: double check this!
        # get qi*qj matrix
        q_ij = qi * qi.transpose(1, 2)
        # remove diagonal elements
        q_ij = torch.gather(q_ij, -1, neighbors) * neighbor_mask

        # compute switch factors
        f = spk.nn.switch_function(r_ij, self.cuton, self.cutoff)

        # compute damped and coulomb components
        if self.lr_cutoff is None:
            coulomb = torch.where(
                neighbor_mask != 0.0, 1 / r_ij, torch.zeros_like(r_ij)
            )
            damped = torch.where(
                neighbor_mask != 0.0,
                1 / (r_ij ** 16 + self.cuton ** 16) ** (1 / 16),
                torch.zeros_like(r_ij),
            )
        else:
            coulomb = torch.where(
                r_ij < self.lr_cutoff,
                1.0 / r_ij + r_ij / self.lr_cutoff ** 2 - 2.0 / self.lr_cutoff,
                torch.zeros_like(r_ij),
            )
            damped = (
                1 / (r_ij ** 16 + self.cuton ** 16) ** (1 / 16)
                + (1 - f) * self.cut_rconstant * r_ij
                - self.cut_constant
            )

        # return sum over all atoms i and neighbors j
        corr_ij = self.ke / 2 * q_ij * (f * damped + (1 - f) * coulomb * neighbor_mask)
        return torch.sum(corr_ij, (-1, -2), keepdim=True).squeeze(-1)


class ZBLRepulsionEnergy(nn.Module):
    def __init__(self, a0=0.5291772105638411, ke=14.399645351950548):
        super(ZBLRepulsionEnergy, self).__init__()
        self.a0 = a0
        self.ke = ke
        self.kehalf = ke / 2
        self.register_parameter("_adiv", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_apow", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c1", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c2", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c3", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c4", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a1", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a2", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a3", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a4", nn.Parameter(torch.Tensor(1)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._adiv, softplus_inverse(1 / (0.8854 * self.a0)))
        nn.init.constant_(self._apow, softplus_inverse(0.23))
        nn.init.constant_(self._c1, softplus_inverse(0.18180))
        nn.init.constant_(self._c2, softplus_inverse(0.50990))
        nn.init.constant_(self._c3, softplus_inverse(0.28020))
        nn.init.constant_(self._c4, softplus_inverse(0.02817))
        nn.init.constant_(self._a1, softplus_inverse(3.20000))
        nn.init.constant_(self._a2, softplus_inverse(0.94230))
        nn.init.constant_(self._a3, softplus_inverse(0.40280))
        nn.init.constant_(self._a4, softplus_inverse(0.20160))

    def forward(self, inputs, atomwise_predictions):
        neighbors = inputs["_neighbors"]
        neighbor_mask = inputs["_neighbor_mask"]
        n_batch, n_atoms, n_neigh = neighbors.shape
        idx_i = torch.tensor([[i for i in range(n_atoms)] for i in range(n_batch)])
        idx_i = idx_i.unsqueeze(-1).expand(n_batch, n_atoms, n_neigh)
        Zf = inputs["_atomic_numbers"].float().unsqueeze(-1)
        r_ij = inputs["distances"]
        z_ex = Zf.expand(n_batch, n_atoms, n_atoms)

        # calculate parameters
        z = z_ex ** F.softplus(self._apow)
        a = (z + z.transpose(1, 2)) * F.softplus(self._adiv)
        # remove diag
        a = torch.gather(a, -1, neighbors) * neighbor_mask

        a1 = F.softplus(self._a1) * a
        a2 = F.softplus(self._a2) * a
        a3 = F.softplus(self._a3) * a
        a4 = F.softplus(self._a4) * a
        c1 = F.softplus(self._c1)
        c2 = F.softplus(self._c2)
        c3 = F.softplus(self._c3)
        c4 = F.softplus(self._c4)
        # normalize c coefficients (to get asymptotically correct behaviour for r -> 0)
        csum = c1 + c2 + c3 + c4
        c1 = c1 / csum
        c2 = c2 / csum
        c3 = c3 / csum
        c4 = c4 / csum
        # actual interactions
        zizj = z_ex * z_ex.transpose(1, 2)
        zizj = torch.gather(zizj, -1, neighbors) * neighbor_mask

        f = (
            c1 * torch.exp(-a1 * r_ij)
            + c2 * torch.exp(-a2 * r_ij)
            + c3 * torch.exp(-a3 * r_ij)
            + c4 * torch.exp(-a4 * r_ij)
        )

        # compute ij values
        corr_ij = torch.where(neighbor_mask!=0, self.kehalf * f * zizj /
                                       r_ij, torch.zeros_like(r_ij))
        return torch.sum(corr_ij, (-1, -2), keepdim=True).squeeze(-1)


class D4DispersionEnergy(nn.Module):
    def __init__(
        self,
        cutoff=None,
        s6=1.00000000,
        s8=1.61679827,
        a1=0.44959224,
        a2=3.35743605,
        g_a=3.0,
        g_c=2.0,
        k2=1.3333333333333333,  # 4/3
        k4=4.10451,
        k5=19.08857,
        k6=254.5553148552,  # 2*11.28174**2
        kn=7.5,
        wf=6.0,
        Zmax=87,
        Bohr=0.5291772105638411,
        Hartree=27.211386024367243,
        dtype=torch.float32,
    ):
        super(D4DispersionEnergy, self).__init__()
        assert Zmax <= 87

        # parameters
        self.register_parameter(
            "_s6", nn.Parameter(softplus_inverse(s6), requires_grad=False)
        )  # s6 is usually not fitted (correct long-range)
        self.register_parameter(
            "_s8", nn.Parameter(softplus_inverse(s8), requires_grad=True)
        )
        self.register_parameter(
            "_a1", nn.Parameter(softplus_inverse(a1), requires_grad=True)
        )
        self.register_parameter(
            "_a2", nn.Parameter(softplus_inverse(a2), requires_grad=True)
        )
        self.register_parameter(
            "_scaleq", nn.Parameter(softplus_inverse(1.0), requires_grad=True)
        )  # parameter to scale charges of reference systems

        # D4 constants
        self.Zmax = Zmax
        self.convert2Bohr = 1 / Bohr
        self.convert2eV = 0.5 * Hartree  # the factor of 0.5 prevents double counting
        self.convert2Angstrom3 = Bohr ** 3
        self.convert2eVAngstrom6 = Hartree * Bohr ** 6
        self.cutoff = cutoff
        if self.cutoff is not None:
            self.cutoff *= self.convert2Bohr
            self.cuton = self.cutoff - Bohr
        self.g_a = g_a
        self.g_c = g_c
        self.k2 = k2
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kn = kn
        self.wf = wf

        # load D4 data
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d4data")
        self.register_buffer(
            "refsys", torch.load(os.path.join(directory, "refsys.pth"))[:Zmax]
        )  # [Zmax,max_nref]
        self.register_buffer(
            "zeff", torch.load(os.path.join(directory, "zeff.pth"))[:Zmax]
        )  # [Zmax]
        self.register_buffer(
            "refh", torch.load(os.path.join(directory, "refh.pth"))[:Zmax]
        )  # [Zmax,max_nref]
        self.register_buffer(
            "sscale", torch.load(os.path.join(directory, "sscale.pth"))
        )  # [18]
        self.register_buffer(
            "secaiw", torch.load(os.path.join(directory, "secaiw.pth"))
        )  # [18,23]
        self.register_buffer(
            "gam", torch.load(os.path.join(directory, "gam.pth"))[:Zmax]
        )  # [Zmax]
        self.register_buffer(
            "ascale", torch.load(os.path.join(directory, "ascale.pth"))[:Zmax]
        )  # [Zmax,max_nref]
        self.register_buffer(
            "alphaiw", torch.load(os.path.join(directory, "alphaiw.pth"))[:Zmax]
        )  # [Zmax,max_nref,23]
        self.register_buffer(
            "hcount", torch.load(os.path.join(directory, "hcount.pth"))[:Zmax]
        )  # [Zmax,max_nref]
        self.register_buffer(
            "casimir_polder_weights",
            torch.load(os.path.join(directory, "casimir_polder_weights.pth"))[:Zmax],
        )  # [23]
        self.register_buffer(
            "rcov", torch.load(os.path.join(directory, "rcov.pth"))[:Zmax]
        )  # [Zmax]
        self.register_buffer(
            "en", torch.load(os.path.join(directory, "en.pth"))[:Zmax]
        )  # [Zmax]
        self.register_buffer(
            "ncount_mask", torch.load(os.path.join(directory, "ncount_mask.pth"))[:Zmax]
        )  # [Zmax,max_nref,max_ncount]
        self.register_buffer(
            "ncount_weight",
            torch.load(os.path.join(directory, "ncount_weight.pth"))[:Zmax],
        )  # [Zmax,max_nref,max_ncount]
        self.register_buffer(
            "cn", torch.load(os.path.join(directory, "cn.pth"))[:Zmax]
        )  # [Zmax,max_nref,max_ncount]
        self.register_buffer(
            "fixgweights", torch.load(os.path.join(directory, "fixgweights.pth"))[:Zmax]
        )  # [Zmax,max_nref]
        self.register_buffer(
            "refq", torch.load(os.path.join(directory, "refq.pth"))[:Zmax]
        )  # [Zmax,max_nref]
        self.register_buffer(
            "sqrt_r4r2", torch.load(os.path.join(directory, "sqrt_r4r2.pth"))[:Zmax]
        )  # [Zmax]
        self.register_buffer(
            "alpha", torch.load(os.path.join(directory, "alpha.pth"))[:Zmax]
        )  # [Zmax,max_nref,23]
        self.max_nref = self.refsys.size(-1)
        self._compute_refc6()

    # the refc6 tensor is rather large and is therefore not stored as a buffer
    # this is used as a workaround
    def refc6(self):
        if self._refc6.dtype != self.cn.dtype or self._refc6.device != self.cn.device:
            self._refc6 = self._refc6.to(dtype=self.cn.dtype, device=self.cn.device)
        return self._refc6

    # important! If reference charges are scaled, this needs to be recomputed every time the parameters change!
    def _compute_refc6(self):
        with torch.no_grad():
            allZ = torch.arange(self.Zmax)
            is_ = self.refsys[allZ, :]
            iz = self.zeff[is_]
            refh = self.refh[allZ, :] * F.softplus(self._scaleq)
            qref = iz
            qmod = iz + refh
            ones_like_qmod = torch.ones_like(qmod)
            qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
            alpha = (
                self.sscale[is_].view(-1, self.max_nref, 1)
                * self.secaiw[is_]
                * torch.where(
                    qmod > 1e-8,
                    torch.exp(
                        self.g_a
                        * (1 - torch.exp(self.gam[is_] * self.g_c * (1 - qref / qmod_)))
                    ),
                    np.exp(self.g_a) * ones_like_qmod,
                ).view(-1, self.max_nref, 1)
            )
            alpha = torch.max(
                self.ascale[allZ, :].view(-1, self.max_nref, 1)
                * (
                    self.alphaiw[allZ, :, :]
                    - self.hcount[allZ, :].view(-1, self.max_nref, 1) * alpha
                ),
                torch.zeros_like(alpha),
            )
            # refc6 is not stored in a buffer because it is quite large and can easily be re-computed
            alpha_expanded = alpha.view(
                alpha.size(0), 1, alpha.size(1), 1, -1
            ) * alpha.view(1, alpha.size(0), 1, alpha.size(1), -1)
            self._refc6 = (
                3
                / np.pi
                * torch.sum(
                    alpha_expanded * self.casimir_polder_weights.view(1, 1, 1, 1, -1),
                    -1,
                )
            )

    def forward(self, inputs, atomwise_predictions, compute_atomic_quantities=False):
        idx_j = inputs["neighbors"]
        n_batch, n_atoms, n_neigh = idx_j.shape
        idx_i = torch.tensor([[i for i in range(n_atoms)] for i in range(n_batch)])
        idx_i.unsqueeze(-1).expand(n_batch, n_atoms, n_neigh)
        Z = inputs["atomic_numbers"]
        r_ij = inputs["distances"]
        qa = atomwise_predictions["qi"]
        
        
        if idx_i.numel() == 0:
            zeros = r_ij.new_zeros(n_atoms)
            return zeros, zeros, zeros
        r_ij = r_ij * self.convert2Bohr  # convert distances to Bohr
        Zi = Z[idx_i]
        Zj = Z[idx_j]
        # calculate coordination numbers
        rco = self.k2 * (self.rcov[Zi] + self.rcov[Zj])
        den = self.k4 * torch.exp(
            -((torch.abs(self.en[Zi] - self.en[Zj]) + self.k5) ** 2) / self.k6
        )
        tmp = den * 0.5 * (1.0 + torch.erf(-self.kn * (r_ij - rco) / rco))
        if self.cutoff is not None:
            tmp = tmp * spk.nn.activations.switch_function(r_ij, self.cuton, self.cutoff)
        covcn = r_ij.new_zeros(n_atoms).index_add_(0, idx_i, tmp)

        # calculate gaussian weights
        gweights = torch.sum(
            self.ncount_mask[Z]
            * torch.exp(
                -self.wf
                * self.ncount_weight[Z]
                * (covcn.view(-1, 1, 1) - self.cn[Z]) ** 2
            ),
            -1,
        )
        norm = torch.sum(gweights, -1, True)
        norm_ = torch.where(
            norm > 1e-8, norm, torch.ones_like(norm)
        )  # necessary, else there can be nans in backprob
        gweights = torch.where(norm > 1e-8, gweights / norm_, self.fixgweights[Z])

        # calculate actual dispersion energy
        iz = self.zeff[Z].view(-1, 1)
        refq = self.refq[Z] * F.softplus(self._scaleq)
        qref = iz + refq
        qmod = iz + qa.view(-1, 1).repeat(1, self.refq.size(1))
        ones_like_qmod = torch.ones_like(qmod)
        qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
        zeta = (
            torch.where(
                qmod > 1e-8,
                torch.exp(
                    self.g_a
                    * (
                        1
                        - torch.exp(
                            self.gam[Z].view(-1, 1) * self.g_c * (1 - qref / qmod_)
                        )
                    )
                ),
                np.exp(self.g_a) * ones_like_qmod,
            )
            * gweights
        )
        zetai = torch.gather(zeta, 0, idx_i.view(-1, 1).repeat(1, zeta.size(1)))
        zetaj = torch.gather(zeta, 0, idx_j.view(-1, 1).repeat(1, zeta.size(1)))
        refc6 = self.refc6()
        refc6ij = refc6[Zi, Zj, :, :]
        zetaij = zetai.view(zetai.size(0), zetai.size(1), 1) * zetaj.view(
            zetaj.size(0), 1, zetaj.size(1)
        )
        c6ij = torch.sum((refc6ij * zetaij).view(refc6ij.size(0), -1), -1)
        sqrt_r4r2ij = np.sqrt(3) * self.sqrt_r4r2[Zi] * self.sqrt_r4r2[Zj]
        a1 = F.softplus(self._a1)
        a2 = F.softplus(self._a2)
        r0 = a1 * sqrt_r4r2ij + a2
        if self.cutoff is None:
            oor6 = 1 / (r_ij ** 6 + r0 ** 6)
            oor8 = 1 / (r_ij ** 8 + r0 ** 8)
        else:
            cut2 = self.cutoff ** 2
            cut6 = cut2 ** 3
            cut8 = cut2 * cut6
            tmp6 = r0 ** 6
            tmp8 = r0 ** 8
            cut6tmp6 = cut6 + tmp6
            cut8tmp8 = cut8 + tmp8
            oor6 = (
                1 / (r_ij ** 6 + tmp6)
                - 1 / cut6tmp6
                + 6 * cut6 / cut6tmp6 ** 2 * (r_ij / self.cutoff - 1)
            )
            oor8 = (
                1 / (r_ij ** 8 + tmp8)
                - 1 / cut8tmp8
                + 8 * cut8 / cut8tmp8 ** 2 * (r_ij / self.cutoff - 1)
            )
            oor6 = torch.where(r_ij < self.cutoff, oor6, torch.zeros_like(oor6))
            oor8 = torch.where(r_ij < self.cutoff, oor8, torch.zeros_like(oor8))
        s6 = F.softplus(self._s6)
        s8 = F.softplus(self._s8)
        edisp = -c6ij * (s6 * oor6 + s8 * sqrt_r4r2ij ** 2 * oor8) * self.convert2eV
        if compute_atomic_quantities:
            alpha = self.alpha[Z, :, 0]
            polarizabilities = torch.sum(zeta * alpha, -1) * self.convert2Angstrom3
            refc6ii = refc6[Z, Z, :, :]
            zetaii = zeta.view(zeta.size(0), zeta.size(1), 1) * zeta.view(
                zeta.size(0), 1, zeta.size(1)
            )
            c6_coefficients = (
                torch.sum((refc6ii * zetaii).view(refc6ii.size(0), -1), -1)
                * self.convert2eVAngstrom6
            )
        else:
            polarizabilities = r_ij.new_zeros(n_atoms)
            c6_coefficients = r_ij.new_zeros(n_atoms)
        return (
            r_ij.new_zeros(n_atoms).index_add_(0, idx_i, edisp),
            polarizabilities,
            c6_coefficients,
        )

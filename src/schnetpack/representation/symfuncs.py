import torch
from torch import nn
from typing import Callable, Dict
from schnetpack import structure
from schnetpack.nn import scatter_add
from tqdm import tqdm

__all__ = ["SymmetryFunctions", "RadialSF", "AngularSF", "AngularSFANI"]


class SymmetryFunctions(nn.Module):
    def __init__(self, radial: nn.Module, angular: nn.Module):
        super(SymmetryFunctions, self).__init__()

        if radial is not None:
            self.n_basis_radial = radial.n_atom_basis
        else:
            self.n_basis_radial = 0

        if angular is not None:
            self.n_basis_angular = angular.n_atom_basis
        else:
            self.n_basis_angular = 0

        self.radial = radial
        self.angular = angular
        self.n_atom_basis = self.n_basis_radial + self.n_basis_angular

        self.register_buffer("symfunc_stddev", torch.ones(1, self.n_atom_basis))
        self.register_buffer("symfunc_mean", torch.zeros(1, self.n_atom_basis))

    def forward(self, inputs: Dict[str, torch.Tensor]):
        atomic_numbers = inputs[structure.Z]
        target_shape = atomic_numbers.shape[0]

        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]

        # Get the neighboring elements
        Zj = atomic_numbers[idx_j]
        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1)

        x = []

        if self.n_basis_radial != 0:
            # Compute contributions of representation
            radial = self.radial(d_ij, Zj)
            # Sum over neighbors
            radial = scatter_add(radial, idx_i, dim_size=target_shape)
            x.append(radial)

        if self.n_basis_angular != 0:
            # Construct everything for computing the angular symmetry functions
            idx_i_triples = inputs[structure.idx_i_triples]
            idx_j_triples = inputs[structure.idx_j_triples]
            idx_k_triples = inputs[structure.idx_k_triples]

            r_ij_angle = r_ij[idx_j_triples]
            r_ik_angle = r_ij[idx_k_triples]

            d_ij_angle = d_ij[idx_j_triples]
            d_ik_angle = d_ij[idx_k_triples]

            Zj_angle = Zj[idx_j_triples]
            Zk_angle = Zj[idx_k_triples]

            # Compute cosine of angle
            cos_theta_ijk = torch.sum(r_ij_angle * r_ik_angle, dim=1) / (
                d_ij_angle * d_ik_angle
            )

            # Compute and accumulate angular representation
            angular = self.angular(
                d_ij_angle, d_ik_angle, Zj_angle, Zk_angle, cos_theta_ijk
            )
            angular = scatter_add(angular, idx_i_triples, dim_size=target_shape)
            x.append(angular)

        x = torch.cat(x, dim=1)

        x = (x - self.symfunc_mean) / self.symfunc_stddev

        return {"scalar_representation": x}

    def standardize(self, data_loader, eps=1e-6):

        with torch.no_grad():
            symfunc_values = []
            for batch in tqdm(data_loader):
                x = self.forward(batch)
                symfunc_values.append(x["scalar_representation"].detach())

            symfunc_values = torch.cat(symfunc_values)

            symfunc_mean = torch.mean(symfunc_values, dim=0, keepdim=True)
            symfunc_stddev = torch.std(symfunc_values, dim=0, keepdim=True)
            symfunc_stddev = torch.where(
                symfunc_stddev < eps, torch.Tensor([1.0]), symfunc_stddev
            )

            self.symfunc_mean = symfunc_mean
            self.symfunc_stddev = symfunc_stddev


class RadialSF(nn.Module):
    def __init__(
        self, radial_basis: nn.Module, elemental_basis: nn.Module, cutoff_fn: Callable
    ):
        super(RadialSF, self).__init__()
        self.n_atom_basis = radial_basis.n_rbf * elemental_basis.n_ebf

        self.radial_basis = radial_basis
        self.elemental_basis = elemental_basis
        self.cutoff_fn = cutoff_fn

    def forward(self, rij: torch.Tensor, Zj: torch.Tensor):
        elemental_weights = self.elemental_basis(Zj)

        radial = self.radial_basis(rij)
        cutoff_ij = self.cutoff_fn(rij)

        radial = radial * cutoff_ij.unsqueeze(-1)
        radial = radial[:, :, None] * elemental_weights[:, None, :]
        radial = radial.view(-1, self.n_atom_basis)

        return radial


class AngularSF(nn.Module):
    def __init__(
        self,
        radial_basis: nn.Module,
        elemental_basis: nn.Module,
        angular_basis: nn.Module,
        cutoff_fn: Callable,
        crossterms: bool = True,
    ):
        super(AngularSF, self).__init__()
        self.n_atom_basis = (
            radial_basis.n_rbf * elemental_basis.n_ebf_combined * angular_basis.n_abf
        )

        self.radial_basis = radial_basis
        self.elemental_basis = elemental_basis
        self.angular_basis = angular_basis
        self.cutoff_fn = cutoff_fn

        self.crossterms = crossterms

    def _radial_part(
        self, rij: torch.Tensor, rik: torch.Tensor, cos_theta_ijk: torch.Tensor
    ):
        # Compute radial contributions (using rjk terms if requested)
        rij_sq = rij * rij
        rik_sq = rik * rik

        radial = rij_sq + rik_sq
        cutoff_ijk = self.cutoff_fn(rij) * self.cutoff_fn(rik)

        if self.crossterms:
            rjk_sq = rij_sq + rik_sq - 2.0 * rij * rik * cos_theta_ijk
            radial = radial + rjk_sq
            cutoff_ijk = cutoff_ijk * self.cutoff_fn(torch.sqrt(rjk_sq))

        radial = self.radial_basis(torch.sqrt(radial))
        radial = radial * cutoff_ijk.unsqueeze(-1)

        return radial

    def forward(
        self,
        rij: torch.Tensor,
        rik: torch.Tensor,
        Zj: torch.Tensor,
        Zk: torch.Tensor,
        cos_theta_ijk: torch.Tensor,
    ):
        # Compute the elemental contributions
        elemental_weights_jk = self.elemental_basis.combine_elements(
            self.elemental_basis(Zj), self.elemental_basis(Zk)
        )

        # Compute radial contributions
        radial_part = self._radial_part(rij, rik, cos_theta_ijk)

        # Compute angular contributions
        angular_part = self.angular_basis(cos_theta_ijk)

        # Combine everything
        angular = (
            angular_part[:, :, None, None]
            * radial_part[:, None, :, None]
            * elemental_weights_jk[:, None, None, :]
        )
        angular = angular.view(-1, self.n_atom_basis)

        return angular


class AngularSFANI(AngularSF):
    def __init__(
        self,
        radial_basis: nn.Module,
        elemental_basis: nn.Module,
        angular_basis: nn.Module,
        cutoff_fn: Callable,
    ):
        super(AngularSFANI, self).__init__(
            radial_basis=radial_basis,
            elemental_basis=elemental_basis,
            angular_basis=angular_basis,
            cutoff_fn=cutoff_fn,
            crossterms=False,
        )

    def _radial_part(
        self, rij: torch.Tensor, rik: torch.Tensor, cos_theta_ijk: torch.Tensor
    ):
        cutoff_ijk = self.cutoff_fn(rij) * self.cutoff_fn(rik)

        radial = self.radial_basis(0.5 * (rij + rik))
        radial = radial * cutoff_ijk.unsqueeze(-1)

        return radial

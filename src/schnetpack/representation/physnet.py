import torch
import torch.nn as nn

from schnetpack.representation.support import Residual, Output, Module, PhysNetCutOff, SSP
from torch.distributions.binomial import Binomial
import schnetpack.properties as structure
import schnetpack.nn as snn

from typing import Callable, Dict

__all__ = ["PhysNet"]

class PhysNet(nn.Module):

    def __init__(
        self,
        n_atom_basis:int,
        n_interactions:int,
        radial_basis : Callable,
        cutoff_fn: Callable,
        activation = SSP,
        n_output_residual = 1,
        n_atomic_residual = 2,
        n_modules = 5,
        max_z = 200
    ):
        super(PhysNet, self).__init__()
        self.rbf = radial_basis
        self.cutoff_fn = cutoff_fn
        self.radial_basis = radial_basis
        
        self.module = nn.ModuleList(
            [Module(n_atom_basis,
                             activation,
                             self.radial_basis.n_rbf,
                             n_interactions,
                             n_output_residual,
                             n_atomic_residual)
             for _ in range(n_modules)
            ]
        )
        
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        
        
    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        idx_m = inputs[structure.idx_m]
        n_atoms = atomic_numbers.shape[0]
        d_ij = torch.norm(r_ij, dim=1)
        num_batch = int(idx_m[-1]) + 1
        
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)
        g_ij = torch.einsum('ij,i->ij', phi_ij, fcut) 
        x = self.embedding(atomic_numbers)
        
        
        
        summation = []
        
        for module in self.module:
            xo, x = module(x, g_ij, idx_i, idx_j, n_atoms)
            summation.append(xo)
        
        
        out = torch.stack(summation)
        
        
        return {"scalar_representation": out}
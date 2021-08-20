import torch
import torch.nn as nn

import schnetpack.properties as properties
import schnetpack.nn as snn

from typing import Callable, Dict
from torch.nn.init import orthogonal_
from torch.nn.init import zeros_


__all__ = ["PhysNet"]

    
class PhysNetResidual(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable, 
    ):
        super(PhysNetResidual, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.linear = snn.Dense(n_atom_basis, 
                                n_atom_basis, 
                                weight_init=orthogonal_, 
                                bias_init=zeros_)
        
        self.sequential = nn.Sequential(
            activation(),
            self.linear,
            activation(),
            self.linear
        )
        
    def forward(self,x):
        return self.sequential(x) + x    
    
    
class PhysNetOutput(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        n_output_residual:int
    ):
        super(PhysNetOutput, self).__init__()
        self.activation = activation()
        
        self.residual = nn.ModuleList(
            [
               PhysNetResidual(n_atom_basis, activation) 
               for _ in range(n_output_residual)
            ]
        )
  
    def forward(self,x):
        
        for module in self.residual:
            x = module(x)
        x = self.activation(x)
        return x
    
class PhysNetModule(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        n_rbf:int,
        n_interactions:int,
        n_output_residual:int,
        n_atomic_residual:int
    ):
        super(PhysNetModule, self).__init__()
        self.activation = activation()
        self.residual = nn.ModuleList(
            [
               PhysNetResidual(n_atom_basis, activation) 
               for _ in range(n_atomic_residual)
            ]
        )
        
        self.interaction= PhysNetInteraction(n_atom_basis, 
                                             activation, 
                                             n_rbf, 
                                             n_interactions)
        
        self.output = PhysNetOutput(n_atom_basis, 
                             activation, 
                             n_output_residual)
        
        
    def forward(
        self,
        x: torch.Tensor,
        g_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int
        
    ):
        
        x = self.interaction(x, g_ij, idx_i, idx_j, n_atoms)
        
        for module in self.residual:
            x = module(x)
        
        return self.output(x), x       
    
class PhysNetInteraction(nn.Module):

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        n_rbf: int,
        n_interactions: int
    ):

        super(PhysNetInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_rbf = n_rbf
        self.activation = activation()
        self.n_interactions = n_interactions
        
        self.linear_f = snn.Dense(n_atom_basis, n_atom_basis, weight_init=orthogonal_, bias_init=zeros_)
        self.linear_g = snn.Dense(n_rbf, n_atom_basis, bias=False, weight_init=orthogonal_)
        
        self.linear_j = snn.Dense(n_atom_basis, 
                                  n_atom_basis, 
                                  activation=activation(),
                                  weight_init=orthogonal_, 
                                  bias_init=zeros_)
        
        self.linear_i = snn.Dense(n_atom_basis, 
                                  n_atom_basis, 
                                  activation=activation(),
                                  weight_init=orthogonal_, 
                                  bias_init=zeros_)
        
        self.residual = nn.ModuleList(
            [
               PhysNetResidual(n_atom_basis, activation) 
               for _ in range(n_interactions)
            ]
        )
        
        self.u = nn.Parameter(torch.ones(n_atom_basis, requires_grad=True))
        
    def forward(
        self,
        x: torch.Tensor,
        g_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int
    ):
        
        x = self.activation(x)
        x_j = x[idx_j]
        xp = self.u*x 
        vp = self.linear_j(x_j)*self.linear_g(g_ij)
        vm = self.linear_i(x)
        v = snn.scatter_add(vp, idx_i, dim_size=n_atoms) + vm
        for module in self.residual:
            v = module(v)
        
        v = self.activation(v)
        
        return xp + self.linear_f(v)
    
class SSP(nn.Module):
    
    def __init__(self, shift = 2.):
        super(SSP, self).__init__()
        self.register_buffer("shift", torch.FloatTensor([shift]))

    def forward(self, inputs: torch.Tensor):
        
        return snn.activations.shifted_softplus(inputs)  
    

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
            [PhysNetModule(n_atom_basis,
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
        
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        idx_m = inputs[properties.idx_m]
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
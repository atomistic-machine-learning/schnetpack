import torch
import torch.nn as nn
import schnetpack.nn as snn
from schnetpack.representation.weight_init import *

from typing import Callable, Dict

import schnetpack.properties as structure

class Residual(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable, 
    ):
        super(Residual, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.linear = nn.Linear(n_atom_basis, n_atom_basis)
        
        ## Initialize
        W_init = torch.Tensor(semi_orthogonal_glorot_weights(n_atom_basis, n_atom_basis, seed=None))
        
        self.linear.weight.data = torch.nn.Parameter(W_init)
        self.linear.bias.data.fill_(0.)
        
        self.sequential = nn.Sequential(
            activation(),
            self.linear,
            activation(),
            self.linear
        )
        
    def forward(self,x):
        return self.sequential(x) + x    
    
    
class Output(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        n_output_residual:int
    ):
        super(Output, self).__init__()
        self.activation = activation()
        
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_output_residual)
            ]
        )
  
    def forward(self,x):
        
        for module in self.residual:
            x = module(x)
        x = self.activation(x)
        return x
    
class Module(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        n_rbf:int,
        n_interactions:int,
        n_output_residual:int,
        n_atomic_residual:int
    ):
        super(Module, self).__init__()
        self.activation = activation()
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_atomic_residual)
            ]
        )
        self.interaction= PhysNetInteraction(n_atom_basis, 
                                             activation, 
                                             n_rbf, 
                                             n_interactions)
        self.output = Output(n_atom_basis, 
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
        
        self.linear_f = nn.Linear(n_atom_basis, n_atom_basis) 
        self.linear_g = nn.Linear(n_rbf, n_atom_basis, bias=False) 
        self.linear_j = nn.Linear(n_atom_basis, n_atom_basis)
        self.linear_i = nn.Linear(n_atom_basis, n_atom_basis)
        
        ## Initialization
        W_init = torch.Tensor(semi_orthogonal_glorot_weights(n_atom_basis, n_atom_basis, seed=None))
        
        self.linear_j.weight.data = torch.nn.Parameter(W_init)
        self.linear_i.weight.data = torch.nn.Parameter(W_init)
        self.linear_f.weight.data = torch.nn.Parameter(W_init)
        self.linear_g.weight.data = torch.nn.Parameter(torch.zeros(self.n_atom_basis, self.n_rbf))
        self.linear_j.bias.data.fill_(0.)
        self.linear_i.bias.data.fill_(0.)
        self.linear_f.bias.data.fill_(0.)
        
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_interactions)
            ]
        )
        self.sequential_i = nn.Sequential( 
                    activation(),
                    self.linear_i,
                    activation()
        )
        self.sequential_j = nn.Sequential(
                    activation(),
                    self.linear_j,
                    activation()
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
        
        x_j = x[idx_j]
        xp = self.u*x ## check this
        vp = self.sequential_j(x_j.float())*self.linear_g(g_ij.float())
        vm = self.sequential_i(x)
        v = snn.scatter_add(vp, idx_i, dim_size=n_atoms) + vm
        for module in self.residual:
            v = module(v)
        
        v = self.activation(v)
        
        return xp + self.linear_f(v)
    
class PhysNetCutOff(nn.Module):
    
    def __init__(self, cutoff: float):
        super(PhysNetCutOff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, d_ij: torch.Tensor):


        # Compute values of cutoff function
        input_cut = 1 - 6*(d_ij/self.cutoff)**5 + 15*(d_ij/self.cutoff)**4 - 10*(d_ij/self.cutoff)**3
        # Remove contributions beyond the cutoff radius
        input_cut *= (d_ij < self.cutoff).float()
        return input_cut
    
class SSP(nn.Module):
    
    def __init__(self, shift = 2.):
        super(SSP, self).__init__()
        self.register_buffer("shift", torch.FloatTensor([shift]))

    def forward(self, inputs: torch.Tensor):
        
        return snn.activations.shifted_softplus(inputs)  
    
def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBF_PhysNet(nn.Module):
    def __init__(self, n_rbf, cutoff):
        super(RBF_PhysNet,self).__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = softplus_inverse(torch.linspace(1.0,np.exp(-cutoff),n_rbf))
        self.centers = snn.activations.shifted_softplus(((centers)))
        
        widths = [softplus_inverse((0.5/((1.0-np.exp(-cutoff))/n_rbf))**2)]*n_rbf
        self.widths = snn.activations.shifted_softplus(torch.Tensor(widths))
    def forward(self, r_ij):
        r_ij = r_ij.unsqueeze(-1)
        g_ij = torch.exp(-self.widths*(torch.exp(-r_ij)-self.centers)**2).cuda()
        
        return g_ij


def _switch_component(x, ones, zeros):
    x_ = torch.where(x <= 0, ones, x)
    return torch.where(x <= 0, zeros, torch.exp(-ones/x_))

def switch_function(x, cuton, cutoff):
    x = (x-cuton)/(cutoff-cuton)
    ones  = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1-x, ones, zeros)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm/(fp+fm)))


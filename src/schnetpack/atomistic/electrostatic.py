import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Dict

def _switch_component(x, ones, zeros):
    x_ = torch.where(x <= 0, ones, x)
    return torch.where(x <= 0, zeros, torch.exp(-ones/x_))

def switch_function(x , cuton: float, cutoff:float):
    x = (x-cuton)/(cutoff-cuton)
    ones  = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1-x, ones, zeros)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm/(fp+fm)))

# def softplus_inverse(x):
#     return x + np.log(-np.expm1(-x))

'''
computes electrostatic energy, switches between a constant value
and the true Coulomb law between cuton and cutoff 
(this is done so that a lr_cutoff does not affect covalent interactions)
'''
class ElectrostaticEnergy(nn.Module):
    def __init__(self, ke: float = 14.399645351950548, cuton:float =2.5, cutoff: float=7.5, lr_cutoff: float =10.):
        super(ElectrostaticEnergy, self).__init__()
        self.ke = ke
        self.kehalf = ke/2
        self.cuton  = cuton
        self.cutoff = cutoff
        self.cuton16 = cuton**16
        self.lr_cutoff = lr_cutoff
        #these are constants for when a lr_cutoff is used
        if lr_cutoff is not None:
            self.cut_rconstant = lr_cutoff**15/(lr_cutoff**16+cuton**16)**(17/16)
            self.cut_constant = 1/(cuton**16+lr_cutoff**16)**(1/16) + lr_cutoff**16/(lr_cutoff**16 + cuton**16)**(17/16)

    def forward(self, N: int, q:torch.Tensor, rij: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor):
        fac = self.kehalf*torch.gather(q, 0, idx_i)*torch.gather(q, 0, idx_j)
        f = switch_function(rij, self.cuton, self.cutoff)
        if self.lr_cutoff is None:
            coulomb = 1/rij
            damped  = 1/(rij**16 + self.cuton16)**(1/16)
        else:
            coulomb = torch.where(rij < self.lr_cutoff, 1.0/rij + rij/self.lr_cutoff**2 - 2.0/self.lr_cutoff, torch.zeros_like(rij))
            damped  = 1/(rij**16 + self.cuton16)**(1/16) + (1-f)*self.cut_rconstant*rij - self.cut_constant
        return q.new_zeros(N, dtype=torch.float).index_add_(0, idx_i, (fac*(f*damped + (1-f)*coulomb)).float()) # .type(torch.float32)



import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.properties as properties
from schnetpack.nn.activations import _switch_component, switch_function
from typing import Callable, Dict
import schnetpack.nn as snn

__all__ = ["ElectrostaticEnergy"]

class ElectrostaticEnergy(nn.Module):
    
    '''
    Computes the electrostatic energy, switches between a constant value
    and the true Coulomb law between cuton and cutoff 
    (this is done so that a lr_cutoff does not affect covalent interactions)
    '''
    
    def __init__(self, ke: float = 14.399645351950548, cuton:float =2.5, cutoff: float=7.5, lr_cutoff: float =10.0, output_key='electrostatic'):
        super(ElectrostaticEnergy, self).__init__()
        self.ke = ke
        self.kehalf = ke/2
        self.cuton  = cuton
        self.cutoff = cutoff
        self.cuton16 = cuton**16
        self.lr_cutoff = lr_cutoff
        self.output_key = output_key
        #these are constants for when a lr_cutoff is used
        if lr_cutoff is not None:
            self.cut_rconstant = lr_cutoff**15/(lr_cutoff**16+cuton**16)**(17/16)
            self.cut_constant = 1/(cuton**16+lr_cutoff**16)**(1/16) + lr_cutoff**16/(lr_cutoff**16 + cuton**16)**(17/16)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        result = {}
        atomic_numbers = inputs[properties.Z]
        q = inputs[properties.partial_charges].squeeze(-1)
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1).cuda()
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        N = atomic_numbers.size(0)
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        
        fac = self.kehalf*torch.gather(q, 0, idx_i)*torch.gather(q, 0, idx_j)
        
        f = switch_function(d_ij, self.cuton, self.cutoff)
        
        if self.lr_cutoff is None:
            coulomb = 1/d_ij
            damped  = 1/(d_ij**16 + self.cuton16)**(1/16)
        else:
            x_coulomb = 1.0/d_ij + d_ij/self.lr_cutoff**2 - 2.0/self.lr_cutoff
            coulomb = torch.where(d_ij < self.lr_cutoff, x_coulomb , torch.zeros_like(d_ij))
            damped  = 1/(d_ij**16 + self.cuton16)**(1/16) + (1-f)*self.cut_rconstant*d_ij - self.cut_constant
        
        y = snn.scatter_add((fac*(f*damped + (1-f)*coulomb)), idx_i, dim_size=N)
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        y = torch.squeeze(y, -1)
        result[self.output_key] = y
        
        return result



import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.properties as properties
from schnetpack.nn.activations import _switch_component, switch_function
from typing import Callable, Dict
import schnetpack.nn as snn
from schnetpack.units import *

__all__ = ["ElectrostaticEnergy"]

class ElectrostaticEnergy(nn.Module):
    
    '''
    Computes the electrostatic energy, switches between a constant value
    and the true Coulomb law between cuton and cutoff 
    (this is done so that a lr_cutoff does not affect covalent interactions)
    '''
    
    def __init__(self, 
                 ke: float = 1.0, 
                 cuton:float = 2.5, 
                 cutoff: float= 7.5, 
                 lr_cutoff: float = 10.0, 
                 output_key='electrostatic',
                 energy_unit = "eV",
                 length_unit = "Ang"
                ):
        super(ElectrostaticEnergy, self).__init__()
        
        self.energy_unit = convert_units("Ha", energy_unit)
        self.length_unit = convert_units("Bohr", length_unit)
        self.conversion_factor = self.energy_unit*self.length_unit
        
        self.ke = ke*self.conversion_factor
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
        
        """ 
        Args:
            
            energy_unit: Sets the unit of energy given by the user. Default is Hartree
            length_unit: Sets the unit of length given by the user. Default is Bohr
            conversion_factor: Conversion factor between atomic units and user-defined units
            ke: Coulomb constant converted to desired units. The default is given in atomic units (Ha and Bohr)
            kehalf: Coulomb constant divided by two
            cuton: Minimum distance from the charge where interaction is computed
            cutoff: Maximum distance from the charge where interaction is computed
            cuton16: Cuton raised to the 16th power
            lr_cutoff: Long range cutoff
            output_key: Key under which the resulting energy is stored
        
        References:
        .. [#Electrostatic Energy] O.Unke, M.Meuwly
        PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges
        https://arxiv.org/abs/1902.08408
        """

    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        result = {}
        atomic_numbers = inputs[properties.Z]
        q = inputs[properties.partial_charges].squeeze(-1)
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1)
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



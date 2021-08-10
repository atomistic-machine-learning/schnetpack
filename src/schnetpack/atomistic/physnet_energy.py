import torch
import torch.nn as nn
import schnetpack.properties as structure

from schnetpack.atomistic.electrostatic import ElectrostaticEnergy
from schnetpack.atomistic.D3 import D4DispersionEnergy
from schnetpack.atomistic.zbl import ZBLRepulsionEnergy
from schnetpack.nn.cutoff import PhysNetCutOff
from typing import Sequence, Union, Callable, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()

class PhysNetEnergy(nn.Module):
    
    def __init__(self, cutoff: float = 10.0):
        super(PhysNetEnergy, self).__init__()
        self.Coulomb = ElectrostaticEnergy(cuton=0.25*cutoff, cutoff=0.75*cutoff, lr_cutoff = cutoff)
        self.E3 = D4DispersionEnergy(cutoff=cutoff)
        self.zbl = ZBLRepulsionEnergy()
        self.cutoff_fn = PhysNetCutOff(10.0)
        
        
    def forward(self, yi: torch.Tensor, inputs: Dict[str, torch.Tensor]):
        
        energy = yi[:,0].cuda()
        charge = yi[:,1].cuda()
        
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        idx_m = inputs[structure.idx_m]
        n_atoms = atomic_numbers.shape[0]
        d_ij = torch.norm(r_ij, dim=1).cuda()
        n_atoms = atomic_numbers.size(0)
        num_batch = int(idx_m[-1]) + 1
        cutoff_values = self.cutoff_fn(d_ij)
        
        Q = torch.zeros(n_atoms,dtype=torch.float).cuda()

        
    
        # Charge Conservation
        Qleftover = Q.index_add(0, idx_m, -charge)
        w = torch.ones(n_atoms,dtype=torch.float).cuda()
        w /= (w.new_zeros(num_batch,dtype=torch.float).index_add_(0, idx_m, w)[idx_m]).cuda()
        qa = charge + torch.gather(Qleftover, 0, idx_m)*w
        
        ec = self.Coulomb(n_atoms ,qa, d_ij, idx_i,  idx_j)
        ea_vdw, _, _  = self.E3(n_atoms, atomic_numbers, qa, d_ij, idx_i, idx_j)
        ez = self.zbl(n_atoms, atomic_numbers, d_ij, cutoff_values, idx_i, idx_j)
        
   

        return (ec+ea_vdw+energy+ez).unsqueeze(-1)
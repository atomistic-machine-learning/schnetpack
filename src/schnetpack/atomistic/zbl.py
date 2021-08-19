import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, Callable, Dict, Optional
from schnetpack.nn.cutoff import PhysNetCutOff
from schnetpack.nn.activations import softplus_inverse
import schnetpack.properties as properties
import schnetpack.nn as snn

__all__ = ["ZBLRepulsionEnergy"]


class ZBLRepulsionEnergy(nn.Module):
    
    '''
    Computes a Ziegler-Biersack-Littmark style repulsion energy
    '''
    
    def __init__(self, 
                 a0: float = 0.5291772105638411, 
                 ke: float =14.399645351950548, 
                 output_key : str = "zbl",
                 trainable : bool = True,
                 reset_param : bool = False
        ):
        super(ZBLRepulsionEnergy, self).__init__()
        self.output_key = output_key
        self.a0 = a0
        self.ke = ke
        self.kehalf = ke/2
        self.cutoff_fn = PhysNetCutOff(10.0)
        
        if trainable:
            self.register_parameter('_adiv', nn.Parameter(torch.tensor(1.)))
            self.register_parameter('_apow', nn.Parameter(torch.tensor(1.)))
            self.register_parameter('a_vector', nn.Parameter(torch.tensor([1.,1.,1.,1.])))
            self.register_parameter('c_vector', nn.Parameter(torch.tensor([1.,1.,1.,1.])))
        else:
            self.register_buffer('_adiv', nn.Parameter(torch.tensor(1.)))
            self.register_buffer('_apow', nn.Parameter(torch.tensor(1.)))
            self.register_buffer('a_vector', nn.Parameter(torch.tensor([1.,1.,1.,1.])))
            self.register_buffer('c_vector', nn.Parameter(torch.tensor([1.,1.,1.,1.])))
            
        if reset_param:
            self.a_vector = nn.Parameter(torch.tensor([3.19980, 0.94229, 0.40290, 0.20162]))
            self.c_vector = nn.Parameter(torch.tensor([0.18175, 0.50986, 0.28022, 0.02817]))
            self.reset_parameters
        
    def reset_parameters(self):
        nn.init.constant_(self._adiv, softplus_inverse(1/(0.8854*self.a0)))
        nn.init.constant_(self._apow, softplus_inverse(0.23))
        

    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        result = {}
        Zf = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1).cuda()
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        N = Zf.size(0)
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        cutoff_values = self.cutoff_fn(d_ij)
        
        z  = Zf**F.softplus(self._apow)
        a  = (z[idx_i] + z[idx_j])*F.softplus(self._adiv)
        a_components = F.softplus(self.a_vector[..., None])*a
        c_components = F.softplus(self.c_vector[..., None])
#         a1 = F.softplus(self._a1)*a
#         a2 = F.softplus(self._a2)*a
#         a3 = F.softplus(self._a3)*a
#         a4 = F.softplus(self._a4)*a
#         c1 = F.softplus(self._c1)
#         c2 = F.softplus(self._c2)
#         c3 = F.softplus(self._c3)
#         c4 = F.softplus(self._c4)
        #normalize c coefficients (to get asymptotically correct behaviour for r -> 0)
        #csum = c1 + c2 + c3 + c4
        c_normed = F.normalize(c_components, p=1, dim=0)
#         c1 = c1/csum
#         c2 = c2/csum
#         c3 = c3/csum
#         c4 = c4/csum
        #actual interactions
        zizj = Zf[idx_i]*Zf[idx_j]
        f_temp = (c_normed*torch.exp(-a_components*d_ij)).sum(0)
        f = f_temp*cutoff_values
        #f = (c1*torch.exp(-a1*d_ij) + c2*torch.exp(-a2*d_ij) + c3*torch.exp(-a3*d_ij) + c4*torch.exp(-a4*d_ij))*cutoff_values
        
        y = snn.scatter_add(self.kehalf*f*zizj/d_ij, idx_i, dim_size=N)
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        y = torch.squeeze(y, -1)
        
        result[self.output_key] = y
    
        return result
    



import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus_inverse(x):
    return x + (torch.log(-torch.expm1(torch.tensor(-x)))).item()

'''
computes a Ziegler-Biersack-Littmark style repulsion energy
'''
class ZBLRepulsionEnergy(nn.Module):
    def __init__(self, a0: float = 0.5291772105638411, ke: float =14.399645351950548):
        super(ZBLRepulsionEnergy, self).__init__()
        self.a0 = a0
        self.ke = ke
        self.kehalf = ke/2
        self.register_parameter('_adiv', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_apow', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_c1', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_c2', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_c3', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_c4', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_a1', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_a2', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_a3', nn.Parameter(torch.tensor(1.)))
        self.register_parameter('_a4', nn.Parameter(torch.tensor(1.)))
        #self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.constant_(self._adiv, softplus_inverse(1/(0.8854*self.a0)))
#         nn.init.constant_(self._apow, softplus_inverse(0.23))
#         nn.init.constant_(self._c1,   softplus_inverse(0.18180))
#         nn.init.constant_(self._c2,   softplus_inverse(0.50990))
#         nn.init.constant_(self._c3,   softplus_inverse(0.28020))
#         nn.init.constant_(self._c4,   softplus_inverse(0.02817))
#         nn.init.constant_(self._a1,   softplus_inverse(3.20000))
#         nn.init.constant_(self._a2,   softplus_inverse(0.94230))
#         nn.init.constant_(self._a3,   softplus_inverse(0.40280))
#         nn.init.constant_(self._a4,   softplus_inverse(0.20160))

    def forward(self, N: int, Zf: torch.Tensor, rij: torch.Tensor, cutoff_values: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor):
        #calculate parameters
        z  = Zf**F.softplus(self._apow)
        a  = (z[idx_i] + z[idx_j])*F.softplus(self._adiv)
        a1 = F.softplus(self._a1)*a
        a2 = F.softplus(self._a2)*a
        a3 = F.softplus(self._a3)*a
        a4 = F.softplus(self._a4)*a
        c1 = F.softplus(self._c1)
        c2 = F.softplus(self._c2)
        c3 = F.softplus(self._c3)
        c4 = F.softplus(self._c4)
        #normalize c coefficients (to get asymptotically correct behaviour for r -> 0)
        csum = c1 + c2 + c3 + c4
        c1 = c1/csum
        c2 = c2/csum
        c3 = c3/csum
        c4 = c4/csum
        #actual interactions
        zizj = Zf[idx_i]*Zf[idx_j]
        f = (c1*torch.exp(-a1*rij) + c2*torch.exp(-a2*rij) + c3*torch.exp(-a3*rij) + c4*torch.exp(-a4*rij))*cutoff_values
        return Zf.new_zeros(N, dtype=torch.float).index_add_(0, idx_i, (self.kehalf*f*zizj/rij).float())



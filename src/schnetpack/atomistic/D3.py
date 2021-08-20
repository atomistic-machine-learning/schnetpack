import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.properties as properties
import math
from typing import Callable, Dict
from schnetpack.nn.activations import _switch_component, switch_function, softplus_inverse
import schnetpack.nn as snn

__all__ = ["D4DispersionEnergy"]




class D4DispersionEnergy(nn.Module):
    
    '''
    Computes D4 dispersion energy with parameters:
    HF      s6=1.00000000, s8=1.61679827, a1=0.44959224, a2=3.35743605
    '''
    
    def __init__(self, 
            cutoff=7.5, 
            s6=1.00000000, 
            s8=1.61679827, 
            a1=0.44959224, 
            a2=3.35743605, 
            g_a=3.0,
            g_c=2.0,
            k2=1.3333333333333333, #4/3
            k4=4.10451,
            k5=19.08857,
            k6=254.5553148552, #2*11.28174**2
            kn=7.5,
            wf=6.0,
            Zmax=87,
            Bohr=0.5291772105638411,
            Hartree=27.211386024367243,
            dtype=torch.float32,
            output_key = "dispersion"
        ):
        super(D4DispersionEnergy, self).__init__()
        assert Zmax <= 87
        
        self.output_key = output_key
        self._s6 = nn.Parameter(torch.tensor([softplus_inverse(s6)]))
        self._s8 = nn.Parameter(torch.tensor([softplus_inverse(s8)]))
        self._a1 = nn.Parameter(torch.tensor([softplus_inverse(a1)]))
        self._a2 = nn.Parameter(torch.tensor([softplus_inverse(a2)]))
        self._scaleq = nn.Parameter(torch.tensor([softplus_inverse(1.0)]))

        #D4 constants
        self.Zmax = Zmax
        self.convert2Bohr = 1/Bohr
        self.convert2eV = 0.5*Hartree #the factor of 0.5 prevents double counting
        self.convert2Angstrom3 = Bohr**3
        self.convert2eVAngstrom6 = Hartree*Bohr**6 
        self.cutoff = cutoff
        self.cuton = 0.25*cutoff
#         if self.cutoff is not None:
#             self.cutoff *= self.convert2Bohr
#             self.cuton = self.cutoff-Bohr
        self.g_a = g_a
        self.g_c = g_c
        self.k2 = k2 
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kn = kn
        self.wf = wf
        
        self.dtype = dtype
        #load D4 data 
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),'d4data')
        self.register_buffer('refsys',torch.load(os.path.join(directory,'refsys.pth'))[:Zmax]) #[Zmax,max_nref]
        self.register_buffer('zeff',torch.load(os.path.join(directory,'zeff.pth'))[:Zmax]) #[Zmax]
        self.register_buffer('refh',torch.load(os.path.join(directory,'refh.pth'))[:Zmax]) #[Zmax,max_nref]
        self.register_buffer('sscale',torch.load(os.path.join(directory,'sscale.pth'))) #[18]
        self.register_buffer('secaiw',torch.load(os.path.join(directory,'secaiw.pth'))) #[18,23]
        self.register_buffer('gam',torch.load(os.path.join(directory,'gam.pth'))[:Zmax]) #[Zmax]
        self.register_buffer('ascale',torch.load(os.path.join(directory,'ascale.pth'))[:Zmax]) #[Zmax,max_nref]
        self.register_buffer('alphaiw',torch.load(os.path.join(directory,'alphaiw.pth'))[:Zmax]) #[Zmax,max_nref,23]
        self.register_buffer('hcount',torch.load(os.path.join(directory,'hcount.pth'))[:Zmax]) #[Zmax,max_nref]
        self.register_buffer('casimir_polder_weights',torch.load(os.path.join(directory,'casimir_polder_weights.pth'))[:Zmax]) #[23]
        self.register_buffer('rcov',torch.load(os.path.join(directory,'rcov.pth'))[:Zmax]) #[Zmax]
        self.register_buffer('en',torch.load(os.path.join(directory,'en.pth'))[:Zmax]) #[Zmax]
        self.register_buffer('ncount_mask',torch.load(os.path.join(directory,'ncount_mask.pth'))[:Zmax]) #[Zmax,max_nref,max_ncount]
        self.register_buffer('ncount_weight',torch.load(os.path.join(directory,'ncount_weight.pth'))[:Zmax]) #[Zmax,max_nref,max_ncount]
        self.register_buffer('cn',torch.load(os.path.join(directory,'cn.pth'))[:Zmax]) #[Zmax,max_nref,max_ncount]
        self.register_buffer('fixgweights',torch.load(os.path.join(directory,'fixgweights.pth'))[:Zmax]) #[Zmax,max_nref]
        self.register_buffer('refq',torch.load(os.path.join(directory,'refq.pth'))[:Zmax]) #[Zmax,max_nref]
        self.register_buffer('sqrt_r4r2',torch.load(os.path.join(directory,'sqrt_r4r2.pth'))[:Zmax]) #[Zmax]
        self.register_buffer('alpha',torch.load(os.path.join(directory,'alpha.pth'))[:Zmax]) #[Zmax,max_nref,23]
        self.max_nref = self.refsys.size(-1)
        self.pi = math.pi
        self._compute_refc6()
        self._cast_to_type()
        
        
    def _cast_to_type(self):
        self.refsys = self.refsys.type(self.dtype)
        self.zeff = self.zeff.type(self.dtype)
        self.refh = self.refh.type(self.dtype)
        self.sscale = self.sscale.type(self.dtype)
        self.secaiw = self.secaiw.type(self.dtype)
        self.gam = self.gam.type(self.dtype)
        self.ascale = self.ascale.type(self.dtype)
        self.alphaiw = self.alphaiw.type(self.dtype)
        self.hcount = self.hcount.type(self.dtype)
        self.casimir_polder_weights = self.casimir_polder_weights.type(self.dtype)
        self.en = self.en.type(self.dtype)
        self.rcov = self.rcov.type(self.dtype)
        self.ncount_mask = self.ncount_mask.type(self.dtype)
        self.ncount_weight = self.ncount_weight.type(self.dtype)
        self.cn = self.cn.type(self.dtype)
        self.fixgweights = self.fixgweights.type(self.dtype)
        self.refq = self.refq.type(self.dtype)
        self.sqrt_r4r2 = self.sqrt_r4r2.type(self.dtype)
        self.alpha = self.alpha.type(self.dtype)
        self._refc6 = self._refc6.type(self.dtype)
        
        
        
        
    #important! If reference charges are scaled, this needs to be recomputed every time the parameters change!
    def _compute_refc6(self):
        with torch.no_grad():
            allZ = torch.arange(self.Zmax)
            is_ = self.refsys[allZ,:]
            iz  = self.zeff[is_]
            refh = self.refh[allZ,:]*F.softplus(self._scaleq)
            qref = iz
            qmod = iz + refh
            ones_like_qmod = torch.ones_like(qmod)
            qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
            alpha = self.sscale[is_].view(-1,self.max_nref,1)*self.secaiw[is_]*torch.where(
                    qmod > 1e-8, torch.exp(self.g_a*(1-torch.exp(self.gam[is_]*self.g_c*(1-qref/qmod_)))), torch.exp(torch.tensor(self.g_a))*ones_like_qmod).view(-1,self.max_nref,1)
            alpha = torch.max(self.ascale[allZ,:].view(-1,self.max_nref,1)*(self.alphaiw[allZ,:,:] 
                    - self.hcount[allZ,:].view(-1,self.max_nref,1)*alpha), torch.zeros_like(alpha))
            #refc6 is not stored in a buffer because it is quite large and can easily be re-computed
            alpha_expanded = alpha.view(alpha.size(0),1,alpha.size(1),1,-1)*alpha.view(1,alpha.size(0),1,alpha.size(1),-1)
            self._refc6 = 3/self.pi*torch.sum(alpha_expanded*self.casimir_polder_weights.view(1,1,1,1,-1),-1)

    def forward(self, inputs: Dict[str, torch.Tensor], compute_atomic_quantities: bool = False):
        
        result = {}
        Z = inputs[properties.Z]
        qa = inputs[properties.partial_charges].squeeze(-1)
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1)
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        N = Z.size(0)
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        
        if idx_i.numel() == 0:
            zeros = d_ij.new_zeros(N)
            return zeros, zeros, zeros
        #convert distances to Bohr
        #d_ij = d_ij*self.convert2Bohr 
        Zi = Z[idx_i]
        Zj = Z[idx_j]
        
        
        #calculate coordination numbers
        rco = self.k2*(self.rcov[Zi] + self.rcov[Zj])
        den = self.k4*torch.exp(-(torch.abs(self.en[Zi]-self.en[Zj])+ self.k5)**2/self.k6)
        tmp = den*0.5*(1.0 + torch.erf(-self.kn*(d_ij-rco)/rco)) 
        if self.cutoff is not None:
            tmp = tmp*switch_function(d_ij, self.cuton, self.cutoff)
        covcn = d_ij.new_zeros(N).index_add_(0, idx_i, tmp)

        #calculate gaussian weights
        gweights = torch.sum(self.ncount_mask[Z]*torch.exp(-self.wf*self.ncount_weight[Z]*
            (covcn.view(-1,1,1)-self.cn[Z])**2),-1)
        norm = torch.sum(gweights,-1,True)
        norm_ = torch.where(norm > 1e-8, norm, torch.ones_like(norm)) #necessary, else there can be nans in backprob
        gweights = torch.where(norm > 1e-8, gweights/norm_, self.fixgweights[Z])

        #calculate actual dispersion energy
        iz = self.zeff[Z].view(-1,1)
        refq = self.refq[Z]*F.softplus(self._scaleq)
        qref = iz + refq
        qmod = iz + qa.view(-1,1).repeat(1,self.refq.size(1))
        ones_like_qmod = torch.ones_like(qmod)
        qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
        zeta = torch.where(qmod > 1e-8, torch.exp(self.g_a*(1-torch.exp(self.gam[Z].view(-1,1)*self.g_c*(1-qref/qmod_)))), torch.exp(torch.tensor(self.g_a))*ones_like_qmod) * gweights 
        zetai = torch.gather(zeta, 0, idx_i.view(-1,1).repeat(1,zeta.size(1)))
        zetaj = torch.gather(zeta, 0, idx_j.view(-1,1).repeat(1,zeta.size(1)))
        refc6 = self._refc6.cuda()
        refc6ij = refc6[Zi,Zj,:,:]
        zetaij = zetai.view(zetai.size(0),zetai.size(1),1)*zetaj.view(zetaj.size(0),1,zetaj.size(1))
        c6ij = torch.sum((refc6ij*zetaij).view(refc6ij.size(0),-1),-1)
        sqrt_r4r2ij = torch.sqrt(torch.tensor(3.))*self.sqrt_r4r2[Zi]*self.sqrt_r4r2[Zj]
        a1 = F.softplus(self._a1)
        a2 = F.softplus(self._a2)
        r0 = a1*sqrt_r4r2ij + a2
        if self.cutoff is None:
            oor6 = 1/(d_ij**6+r0**6)
            oor8 = 1/(d_ij**8+r0**8)
        else:
            cut2 = self.cutoff**2
            cut6 = cut2**3
            cut8 = cut2*cut6
            tmp6 = r0**6
            tmp8 = r0**8
            cut6tmp6 = cut6 + tmp6
            cut8tmp8 = cut8 + tmp8
            oor6 = 1/(d_ij**6+tmp6) - 1/cut6tmp6 + 6*cut6/cut6tmp6**2 * (d_ij/self.cutoff-1)
            oor8 = 1/(d_ij**8+tmp8) - 1/cut8tmp8 + 8*cut8/cut8tmp8**2 * (d_ij/self.cutoff-1)
            oor6 = torch.where(d_ij < self.cutoff, oor6, torch.zeros_like(oor6))
            oor8 = torch.where(d_ij < self.cutoff, oor8, torch.zeros_like(oor8))
        s6 = F.softplus(self._s6)
        s8 = F.softplus(self._s8)
        edisp = -c6ij*(s6*oor6 + s8*sqrt_r4r2ij**2*oor8)*self.convert2eV
        
        if compute_atomic_quantities:
            alpha   = self.alpha[Z,:,0]
            polarizabilities = torch.sum(zeta * alpha,-1)*self.convert2Angstrom3
            refc6ii = refc6[Z,Z,:,:]
            zetaii = zeta.view(zeta.size(0),zeta.size(1),1)*zeta.view(zeta.size(0),1,zeta.size(1))
            c6_coefficients  = torch.sum((refc6ii*zetaii).view(refc6ii.size(0),-1),-1)*self.convert2eVAngstrom6
        else:
            polarizabilities = d_ij.new_zeros(N)
            c6_coefficients  = d_ij.new_zeros(N)
        
        y = snn.scatter_add(edisp, idx_i, dim_size=N) 
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        y = torch.squeeze(y, -1)
        result[self.output_key] = y
        
        return result
        
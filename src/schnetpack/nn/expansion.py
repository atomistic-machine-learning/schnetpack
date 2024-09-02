import torch
import torch.nn as nn

__all__ = ["SphcBasisExpansion"]

class SphcBasisExpansion(nn.Module):
    
    def __init__(self, sphc_basis_expansion_fn: nn.Module, sphc_cutoff_fn: nn.Module):
        super(SphcBasisExpansion, self).__init__()
        self.sphc_basis_expansion_fn = sphc_basis_expansion_fn
        self.sphc_cutoff_fn = sphc_cutoff_fn
        
    def forward(self, inputs: torch.Tensor):
        m_cut_ij = self.sphc_cutoff_fn(inputs)
        for col_idx in range(inputs.shape[1]):
            # get the corresponding column of chi
            chi_l = inputs[:, col_idx]
            exp_chi_l = self.sphc_basis_expansion_fn(chi_l)
            exp_chi_l = torch.where(m_cut_ij != 0, exp_chi_l * m_cut_ij, 0) # shape: (n_pairs,n_rbfs)
        return exp_chi_l
        
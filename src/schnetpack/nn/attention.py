from typing import Callable, Dict, Union, Tuple, Sequence, Optional, Any
from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F
import functorch
from functorch import combine_state_for_ensemble
import schnetpack.nn as snn
from schnetpack.nn.utils import equal_head_split 

__all__ = [
    "DummyIdentity",
    "ConvAttention", 
    "SphConvAttention", 
    "AttentionAggregation", 
    "ConvAttentionCoefficients",
    "VmapConvAttention",
    "VmapSphConvAttention"
    ]



class DummyIdentity(nn.Module):
    r"""A placeholder identity operator
     for attention aggregation in spherical convolution attention layer.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
            self, 
            x: torch.Tensor,
            alpha_ij:torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:
        return alpha_ij

class AttentionAggregation(nn.Module):
    
    def __init__(self,num_features):
        super().__init__()
        self.v_j_linear = nn.Linear(num_features,num_features,bias=False)

    def forward(
            self, 
            x: torch.Tensor,
            alpha_ij: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: atomic embeddings, shape: (n,F)
            alpha_ij: attention coefficients, shape: (n_pairs)
            idx_i: index centering atom, shape: (n_pairs)
            idx_j: index neighboring atom, shape: (n_pairs)

        """
        v_j = self.v_j_linear(x)[idx_j] # shape: (n_pairs,F)
        y = snn.scatter_add(alpha_ij[:,None] * v_j, idx_i, dim_size=x.shape[0]) # shape: (n,F)
        return y

class ConvAttentionCoefficients(nn.Module):
    
    def __init__(self,num_features):
        super().__init__()

        self.q_i_layer = nn.Linear(num_features,num_features,bias=False)
        self.k_j_layer = nn.Linear(num_features,num_features,bias=False) 
    
    def forward(
            self,
            x: torch.Tensor,
            w_ij: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:

        """
        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """

        q_i = self.q_i_layer(x)[idx_i]#[:,idx_i,:] 
        k_j = self.k_j_layer(x)[idx_j]  # shape: (n_pairs,F)
        out = (q_i * w_ij * k_j).sum(axis=-1) / math.sqrt(x.shape[-1])
        return out



class VmapConvAttention(nn.Module):

    def __init__(self,nheads,num_features):
        super().__init__()

        self.nheads = nheads
        self.num_features = num_features // nheads
        self.coeff_fn = nn.ModuleList(
                [ConvAttentionCoefficients(self.num_features) for _ in range(self.nheads)])

        self.aggregate_fn = nn.ModuleList(
            [AttentionAggregation(self.num_features) for _ in range(self.nheads)]
            )

    def vec_calc(self, nn, x_heads, w_heads, idx_i, idx_j):
        '''to vectorize the head calculations, resulting in speed up'''
        fmodel, params, buffers = combine_state_for_ensemble(nn)
        value =  functorch.vmap(
            fmodel, (0,0,0,0,None, None),out_dims=1)(
                params, 
                buffers, 
                x_heads,
                w_heads,
                idx_i,idx_j)
        return value

    def forward(self,x,w_ij,idx_i,idx_j,phi_r_cut):

        # first view in block structure and then permutating it
        x_heads = x.view(x.shape[0], self.nheads, self.num_features).permute(1, 0, 2).contiguous()
        w_heads = w_ij.view(w_ij.shape[0], self.nheads, self.num_features).permute(1, 0, 2).contiguous() 

        # calculate alpha and aggregate for every head
        alpha = self.vec_calc(self.coeff_fn, x_heads, w_heads, idx_i, idx_j) * phi_r_cut#[:,None]
        # permutation necessary for using the aggregation function
        alpha = alpha.permute(1,0).contiguous()
        out = self.vec_calc(self.aggregate_fn, x_heads, alpha, idx_i, idx_j)
        x_ = out.permute(1,0,2).contiguous().view(x.shape[0], -1)

        return x_


class VmapSphConvAttention(nn.Module):

    r"""the number of heads in the SPHC update equals the
    number of degrees in the SPHC vector 
    the vecorized version of the SphConvAttention layer"""
    def __init__(
            self,
            num_features: int,
            degrees: Sequence[int], # degrees
            aggregate_fn: Union[Callable, nn.Module] = None): 
        
        super().__init__()
        self.nheads = len(list(degrees))
        self.harmonic_orders = degrees
        self.record = {}
        self.aggregate_fn = aggregate_fn
        self.num_features = num_features // self.nheads
        self.register_buffer("degrees", torch.LongTensor(degrees))
        self.register_buffer("repeats", torch.LongTensor([2 * y + 1 for y in list(self.degrees)]))
        
        # more concise structure and allows to use aggregation function as used in feature block if needed
        if self.aggregate_fn is None:
            self.aggregate_fn = nn.ModuleList([DummyIdentity(self.num_features) for _ in range(self.nheads)])

        self.coeff_fn = nn.ModuleList(
            [ConvAttentionCoefficients(self.num_features) for _ in range(self.nheads)]
        )

    def vec_calc(self, nn, x_heads, w_heads, idx_i, idx_j):

        fmodel, params, buffers = combine_state_for_ensemble(nn)
        value =  functorch.vmap(
            fmodel, (0,0,0,0,None, None),out_dims=1)(
                params, 
                buffers, 
                x_heads,
                w_heads,
                idx_i,idx_j)
        
        return value

    def forward(
            self,
            chi:torch.Tensor,
            sph_ij:torch.Tensor,
            x: torch.Tensor,
            w_ij: torch.Tensor,
            idx_i: torch.Tensor,
            phi_r_cut: torch.Tensor,
            phi_chi_cut: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:
        

        # first view in block structure and then permutating it
        x_heads = x.view(x.shape[0], self.nheads, self.num_features).permute(1, 0, 2) # shape: (n_heads,n,F_head)
        w_heads = w_ij.view(w_ij.shape[0], self.nheads, self.num_features).permute(1, 0, 2)  # shape (n_heads,n_pairs,F_head)

        alpha_ij = self.vec_calc(self.coeff_fn, x_heads, w_heads, idx_i, idx_j)
        alpha_r_ij = (alpha_ij * phi_r_cut)#.permute(1,0).contiguous()
        alpha_s_ij = (alpha_ij * phi_chi_cut[:,None])#.permute(1,0).contiguous()

        alpha_ij = alpha_r_ij + alpha_s_ij
        self.record["alpha"] = alpha_ij

        alpha_ij = torch.repeat_interleave(alpha_ij,self.repeats,dim=-1) # shape: (n_pairs,m_tot)
        chi_ = snn.scatter_add(alpha_ij * sph_ij, idx_i, dim_size=x.shape[0]) # shape: (n,m_tot)

        return chi_


class ConvAttention(nn.Module):

    """slow implementation of ConvAttention since not vectorized"""
    def __init__(
            self,
            num_features:int,
            nheads:int = 2,
            debug_tag: str = None):

        super().__init__()
        self.nheads = nheads
        self.num_features = num_features // self.nheads
        self.record = {}
        
        # every layer consists of n heads
        self.coeff_fn = nn.ModuleList(
            [ConvAttentionCoefficients(self.num_features) for _ in range(self.nheads)]
        )

        self.aggregate_fn = nn.ModuleList(
            [AttentionAggregation(self.num_features) for _ in range(self.nheads)]
            )

    def forward(
            self,
            x: torch.Tensor,
            w_ij: torch.Tensor,
            phi_r_cut: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:
        
        """

        Args:
            x:atomic embeddings, shape: (n,F)
            w_ij: filter, shape: (n_pairs,F)
            phi_r_cut: cutoff that scales attention coefficients, shape: (n_pairs)

        Returns:

        """

        # split input into heads is equal to ref, because head_so3krates - head_here is zero
        original_shape = x.shape
        inv_x_head_split, x_heads = equal_head_split(x, n_heads=self.nheads)  # shape: (n,n_heads,F_head)
        _, w_heads = equal_head_split(w_ij, n_heads=self.nheads)  # shape: (n_pairs,n_heads,F_head)

        results = []
        alphas = []
        for head in range(self.nheads):
            # TODO make phi_r_cut shape consistent should only be (n_pairs)
            alpha = self.coeff_fn[head](x_heads[:,head,:], w_heads[:,head,:], idx_i, idx_j) * phi_r_cut[:,0]
            alphas.append(alpha)
        alpha_compare = torch.stack(alphas,dim=1)

        # self.record["alpha"]  = alpha_compare
        for head in range(self.nheads):
            res = self.aggregate_fn[head](x_heads[:,head,:], alpha_compare[:,head], idx_i, idx_j)
            results.append(res)
            # # saving the attention weights for later analysis

        results_compare = torch.stack(results,dim=1)
        self.record["alpha"] = alpha_compare
        x_ = inv_x_head_split(results_compare)  # shape: (n_heads,n_pairs)
        return x_  # shape: (n_heads, n_pairs)



class SphConvAttention(nn.Module):

    r"""the number of heads in the SPHC update equals the
    number of degrees in the SPHC vector  """
    def __init__(
            self,
            num_features: int,
            degrees: Sequence[int], # degrees
            aggregate_fn: Union[Callable, nn.Module] = None,
            debug_tag: str = None): 
        
        super().__init__()
        self.nheads = len(list(degrees))
        self.harmonic_orders = degrees
        self.record = {}
        self.aggregate_fn = aggregate_fn
        self.num_features = num_features // self.nheads
        self.register_buffer("degrees", torch.LongTensor(degrees))
        self.register_buffer("repeats", torch.LongTensor([2 * y + 1 for y in list(self.degrees)]))
        # more concise structure and allows to use aggregation function as used in feature block if needed
        if self.aggregate_fn is None:
            self.aggregate_fn = nn.ModuleList([DummyIdentity(self.num_features) for _ in range(self.nheads)])


        self.coeff_fn = nn.ModuleList(
            [ConvAttentionCoefficients(self.num_features) for _ in range(self.nheads)]
        )


    def forward(
            self,
            chi:torch.Tensor,
            sph_ij:torch.Tensor,
            x: torch.Tensor,
            w_ij: torch.Tensor,
            idx_i: torch.Tensor,
            phi_r_cut: torch.Tensor,
            phi_chi_cut: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:
        


        inv_x_head_split, x_heads = equal_head_split(x, n_heads=self.nheads)  # shape: (n,n_heads,F_head)
        _, w_ij_heads = equal_head_split(w_ij, n_heads=self.nheads)  # shape: (n_pairs,n_heads,F_head)

        # Apply each head's ConvAttentionCoefficients and stack results
        # which is super slow
        results = []
        all_alphas_r_ij,all_alphas_s_ij,all_alphas = ([],[],[]) 

        for head in range(self.nheads):
            # TODO make phi_r_cut shape consistent should only be (n_pairs)
            alpha = self.coeff_fn[head](x_heads[:,head,:], w_ij_heads[:,head,:], idx_i, idx_j)
            alpha_r_ij = alpha * phi_r_cut[:,0]
            alpha_s_ij = alpha * phi_chi_cut #  make consistent shape  ist shape (n_pairs,) soll shape (n_pairs,)
            all_alphas_r_ij.append(alpha_r_ij)
            all_alphas_s_ij.append(alpha_s_ij)
            all_alphas.append(alpha_r_ij + alpha_s_ij)

        # alphas sind correct to ref
        #alpha_compare_r_ij = torch.stack(all_alphas_r_ij,dim=1)
        alpha_ij = torch.stack(all_alphas,dim=1)
        self.record["alpha"] = alpha_ij

        # gleich wie ref methode
        alpha_ij = torch.repeat_interleave(alpha_ij,self.repeats,dim=-1) # shape: (n_pairs,m_tot)
        # gleich wie ref methode
        chi_ = snn.scatter_add(alpha_ij * sph_ij, idx_i, dim_size=x.shape[0]) # shape: (n,m_tot)

        return chi_
    
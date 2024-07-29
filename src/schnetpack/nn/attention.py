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
        self.register_parameter("help_device", torch.nn.Parameter((torch.tensor([1.]))))
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
        #vx = torch._C._functorch.get_unwrapped(v_j)
        v_j_1 = alpha_ij[:,None] * v_j
        # v_j_1_wrap = torch._C._functorch.get_unwrapped(v_j_1)
        y = snn.scatter_add(v_j_1, idx_i, dim_size=x.shape[0]) # shape: (n,F)
        return y
        #y = snn.scatter_add(alpha_ij[:,None] * v_j, idx_i, dim_size=x.shape[0]) # shape: (n,F)
        #return y

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
        self.record = {}
        self.nheads = nheads
        self.register_buffer("help_device", torch.LongTensor((nheads)))
        self.num_features = num_features // nheads
        self.coeff_fn = nn.ModuleList(
                [ConvAttentionCoefficients(self.num_features) for _ in range(self.nheads)])

        self.aggregate_fn = nn.ModuleList(
            [AttentionAggregation(self.num_features) for _ in range(self.nheads)]
            )

        #self.reset_parameters()

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

    def helper_params_read_in(self,data,level,device):

        """this is just a helper function for debugging purposes"""
        weights,bias = (data[level]["weights"].T,data[level]["bias"])

        if len(weights.shape) == 1:
            weights = weights[:,None]

        weights = torch.nn.Parameter(torch.tensor(weights))#,device = device)
        if bias is not None:
            bias = torch.nn.Parameter(torch.tensor(bias))#,device = device)
        return weights,bias

    def reset_parameters(self):
        import numpy as np
        path = "/home/elron/phd/projects/ba_betreuung/data/params.npz"
        data = np.load(path,allow_pickle=True)
        feature_block = data["layers_0"].item()['FeatureBlock_0']
        levels = list(feature_block.keys())
        device = self.help_device.device

        attention_levels = levels[4:6]
        weights_A, bias_A = self.helper_params_read_in(feature_block,attention_levels[0],device)
        weights_B, bias_B = self.helper_params_read_in(feature_block,attention_levels[1],device)

        self.coeff_fn[0].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,0])
        self.coeff_fn[0].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,0])

        self.coeff_fn[1].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,1])
        self.coeff_fn[1].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,1])

        attention_agg_levels = levels[6:8]
        for n,level in enumerate(attention_agg_levels):
            weights, bias = self.helper_params_read_in(feature_block,level,device)
            # Ansatz: first layer und second layer aggregation sieht aus wie ref wenn weights.T loaded
            self.aggregate_fn[0].v_j_linear.weight = torch.nn.Parameter(weights[:,:,0])
            self.aggregate_fn[1].v_j_linear.weight = torch.nn.Parameter(weights[:,:,1])

    def helper(self,x):

        import os
        import numpy as np

        ref_x =  np.load(
                '/home/elron/phd/projects/ba_betreuung/data/x_feature_block_convattention.npy')
        ref_x_heads = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/x_heads_feature_block_convattention.npy',
                )
        ref_w =  np.load(
                '/home/elron/phd/projects/ba_betreuung/data/w_feature_block_convattention.npy',
                )
        
        ref_w_heads = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/w_heads_feature_block_convattention.npy',
        )
        ref_alpha = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/alpha_feature_block_convattention.npy',
        )
        ref_idx_i = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/idx_i_feature_block_convattention.npy',
        )
        ref_idx_j = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/idx_j_feature_block_convattention.npy',
        )
        ref_phi_r_cut = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/phi_r_cut_feature_block_convattention.npy',
                )
        ref_alpha_cut = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/alpha_cut_feature_block_convattention.npy',
                )
        
        ref_out = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/out_feature_block_convattention.npy',
                )
        ref_x_ = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/agg_alpha_feature_block_convattention.npy',
                )

        ref_v_j = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/v_j_feature_block_convattention.npy',
                )
        ref_v_j_1 = np.load(
                '/home/elron/phd/projects/ba_betreuung/data/v_j_1_feature_block_convattention.npy',
                )

        ref_x = torch.tensor(ref_x,device=x.device).squeeze()
        ref_x_heads = torch.tensor(ref_x_heads,device=x.device).squeeze()
        ref_w = torch.tensor(ref_w,device=x.device).squeeze()
        ref_w_heads = torch.tensor(ref_w_heads,device=x.device).squeeze()
        ref_alpha = torch.tensor(ref_alpha,device=x.device).squeeze()
        ref_idx_i = torch.tensor(ref_idx_i,device=x.device).squeeze()
        ref_idx_j = torch.tensor(ref_idx_j,device=x.device).squeeze()
        ref_phi_r_cut = torch.tensor(ref_phi_r_cut,device=x.device).squeeze()
        ref_alpha_cut = torch.tensor(ref_alpha_cut,device=x.device).squeeze()
        ref_out = torch.tensor(ref_out,device=x.device).squeeze()
        ref_x_ = torch.tensor(ref_x_,device=x.device).squeeze()
        ref_v_j = torch.tensor(ref_v_j,device=x.device).squeeze()
        ref_v_j_1 = torch.tensor(ref_v_j_1,device=x.device).squeeze()

        pairmask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32,device=x.device)

        return (ref_x,
                ref_x_heads,
                ref_w,
                ref_w_heads,
                ref_alpha,
                ref_idx_i,
                ref_idx_j,
                ref_phi_r_cut,
                ref_alpha_cut,
                ref_out,
                ref_x_,
                ref_v_j,
                ref_v_j_1,
                pairmask)

    def debug(self,x,w_ij,idx_i,idx_j,phi_r_cut):

        (   ref_x,
            ref_x_heads,
            ref_w,
            ref_w_heads,
            ref_alpha,
            ref_idx_i,
            ref_idx_j,
            ref_phi_r_cut,
            ref_alpha_cut,
            ref_out,
            ref_x_,
            ref_v_j,
            ref_v_j_1,
            pairmask) = self.helper(x)
        ##### --------- TEST ATTENTION  #######
        ### resultat: die vmap ist identisch zu ref so3krates
        """
        0. [x] selbes device wie ref benutzt (cpu cuda numerical diffs)
        1. [x] selbe weights benutzt wie ref
        2. [x] selbe idx_i und idx_j benutzt wie ref
        3. [x] ref x umgeformt zu heads wie in unserer implementierung
        4. [x] attention coeff mit umgeformten head berechnet --> test alpha
        5. [x] ref_alpha == test_alpha
        """
        test_x_heads = ref_x.view(ref_x.shape[0], self.nheads, self.num_features).permute(1, 0, 2).contiguous()
        test_w_heads = ref_w.view(ref_w.shape[0], self.nheads, self.num_features).permute(1, 0, 2).contiguous() 
        test_alpha = self.vec_calc(self.coeff_fn, test_x_heads, test_w_heads, idx_i, idx_j) * ref_phi_r_cut[:,None]
        #test_alpha_2 = self.vec_calc(self.coeff_fn, test_x_heads, test_w_heads, idx_i, idx_j) * ref_phi_r_cut[:,None]
        #### ------ TEST ATTENTION AGGREGATION ####
        """ warum ist phi_r_cut nicht gleich, checken!!!
        1. [x] selbe weights benutzt wie ref
        2. [x] selbe idx_i und idx_j benutzt wie ref
        3. [x] ref alpha * v_j ist richtig
        3. [x] test_alpha benutzt und test_x_heads benutzt
        4. [x] geklärt was der sinn von pair mask ist
        
        """
        test_alpha = test_alpha.permute(1,0).contiguous()

        # nur weil beim radial mitgeschleppt wird zeros, aber wir sollten das nicht haben
        ref_tmp_idx_i = torch.tensor([ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  5,  5,  6,
         6,  7,  7,  8,  8,  9,  9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12,
        0, 0, 0, 0, 0, 0, 0, 0],device=x.device)
        test_v_j_1= self.vec_calc(self.aggregate_fn, test_x_heads, test_alpha, ref_idx_i, ref_idx_j)
        yA = self.vec_calc(self.aggregate_fn, test_x_heads, test_alpha, ref_idx_i, ref_idx_j)
        # wenn test out alpha * v_j returned
        # (test_v_j_1.permute(1,0,2) != ref_v_j_1).sum() ist 0 also korrekt
        # yA == ref_out
        # yA_test equal to ref_x_
        yA_test = yA.contiguous().view(x.shape[0], -1)

        # yB = snn.scatter_add(
        #     test_v_j_1, 
        #     ref_idx_i, 
        #     dim_size=x.shape[0],
        #     dim=0) # shape: (n,F)
        
        # test_x_ = yB.contiguous().view(x.shape[0], -1)
        #test_x_ = test_out.permute(1,0,2).contiguous().view(x.shape[0], -1)
        print("Done")

    def forward(self,x,w_ij,idx_i,idx_j,phi_r_cut):

        # first view in block structure and then permutating it
        x_heads = x.view(x.shape[0], self.nheads, self.num_features).permute(1, 0, 2).contiguous() # shape: (n_heads,n_atoms,F_head)
        w_heads = w_ij.view(w_ij.shape[0], self.nheads, self.num_features).permute(1, 0, 2).contiguous() 

        # calculate alpha and aggregate for every head
        alpha = self.vec_calc(self.coeff_fn, x_heads, w_heads, idx_i, idx_j) * phi_r_cut  #[:,None]
        # permutation necessary for using the aggregation function
        alpha = alpha.permute(1,0).contiguous()
        self.record["alpha"] = alpha
        y = self.vec_calc(self.aggregate_fn, x_heads, alpha, idx_i, idx_j)
        x_ = y.contiguous().view(x.shape[0], -1)
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
        self.register_buffer("help_device", torch.LongTensor((self.nheads)))
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

        #self.reset_parameters()

    def helper_params_read_in(self,data,level,device):

        """this is just a helper function for debugging purposes"""
        weights,bias = (data[level]["weights"].T,data[level]["bias"])

        if len(weights.shape) == 1:
            weights = weights[:,None]

        weights = torch.nn.Parameter(torch.tensor(weights))#,device = device)
        if bias is not None:
            bias = torch.nn.Parameter(torch.tensor(bias))#,device = device)
        return weights,bias

    def reset_parameters(self):
        import numpy as np
        path = "/home/elron/phd/projects/ba_betreuung/data/params.npz"
        data = np.load(path,allow_pickle=True)
        feature_block = data["layers_0"].item()['GeometricBlock_0']
        levels = list(feature_block.keys())
        device = self.help_device.device

        attention_levels = levels[4:6]
        weights_A, bias_A = self.helper_params_read_in(feature_block,attention_levels[0],device)
        weights_B, bias_B = self.helper_params_read_in(feature_block,attention_levels[1],device)

        self.coeff_fn[0].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:])
        self.coeff_fn[0].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:])

        #self.coeff_fn[1].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,1])
        #self.coeff_fn[1].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,1])

        # attention_agg_levels = levels[6:8]
        # for n,level in enumerate(attention_agg_levels):
        #     weights, bias = self.helper_params_read_in(feature_block,level,device)
        #     # Ansatz: first layer und second layer aggregation sieht aus wie ref wenn weights.T loaded
        #     self.aggregate_fn[0].v_j_linear.weight = torch.nn.Parameter(weights[:,:,0])
        #     self.aggregate_fn[1].v_j_linear.weight = torch.nn.Parameter(weights[:,:,1])

    def helper(self,x):

        import os
        import numpy as np

        file_paths = {
            'x': '/home/elron/phd/projects/ba_betreuung/data/x_geometric_block_convattention.npy',
            'x_heads': '/home/elron/phd/projects/ba_betreuung/data/x_heads_geometric_block_convattention.npy',
            'w': '/home/elron/phd/projects/ba_betreuung/data/w_geometric_block_convattention.npy',
            'w_heads': '/home/elron/phd/projects/ba_betreuung/data/w_heads_geometric_block_convattention.npy',
            
            'alpha': '/home/elron/phd/projects/ba_betreuung/data/alpha_geometric_block_convattention.npy',
            'alpha_r_ij': '/home/elron/phd/projects/ba_betreuung/data/alpha_r_ij_geometric_block_convattention.npy',
            'alpha_ij_repeat': '/home/elron/phd/projects/ba_betreuung/data/alpha_ij_repeat_geometric_block_convattention.npy',
            'alpha_ij_2': '/home/elron/phd/projects/ba_betreuung/data/alpha_ij_2_geometric_block_convattention.npy',
            
            'idx_i': '/home/elron/phd/projects/ba_betreuung/data/idx_i_geometric_block_convattention.npy',
            'idx_j': '/home/elron/phd/projects/ba_betreuung/data/idx_j_geometric_block_convattention.npy',
            
            'phi_r_cut': '/home/elron/phd/projects/ba_betreuung/data/phi_r_cut_geometric_block_convattention.npy',
            
            'out': '/home/elron/phd/projects/ba_betreuung/data/out_geometric_block_convattention.npy',
            'chi': '/home/elron/phd/projects/ba_betreuung/data/chi_geometric_block_convattention.npy',
        }

        data_dict = {}
        for key, file_path in file_paths.items():
            data_dict[key] = np.load(file_path)

        data_dict = {key: torch.tensor(value, device=x.device).squeeze() for key, value in data_dict.items()}
        pairmask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32, device=x.device)

        data_dict['pairmask'] = pairmask

        return data_dict

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

    def debug(self,
            chi:torch.Tensor,
            sph_ij:torch.Tensor,
            x: torch.Tensor,
            w_ij: torch.Tensor,
            idx_i: torch.Tensor,
            phi_r_cut: torch.Tensor,
            phi_chi_cut: torch.Tensor,
            idx_j: torch.Tensor) -> torch.Tensor:
        
        ref_data = self.helper(x)
        (ref_x, 
            ref_x_heads, 
            ref_w_ij, 
            ref_w_heads, 
            ref_alpha, 
            ref_alpha_r_ij, 
            ref_alpha_ij_repeat,
            ref_alpha_ij_2, 
            ref_idx_i, 
            ref_idx_j, 
            ref_phi_r_cut, 
            ref_out, 
            ref_chi, 
            pairmask) = ref_data.values()

        # first view in block structure and then permutating it
        test_x_heads = ref_x.view(ref_x.shape[0], self.nheads, self.num_features).permute(1, 0, 2) # shape: (n_heads,n,F_head)
        test_w_heads = ref_w_ij.view(ref_w_ij.shape[0], self.nheads, self.num_features).permute(1, 0, 2)  # shape (n_heads,n_pairs,F_head)

        test_alpha_ij = self.vec_calc(self.coeff_fn, test_x_heads, test_w_heads, idx_i, idx_j)
        test_alpha_r_ij = test_alpha_ij * ref_phi_r_cut[:,None] #.permute(1,0).contiguous()
        test_alpha_ij_repeat = torch.repeat_interleave(test_alpha_r_ij,self.repeats,dim=-1) # shape: (n_pairs,m_tot)
        test_chi_ = snn.scatter_add(test_alpha_ij_repeat * sph_ij, idx_i, dim_size=x.shape[0]) # shape: (n,m_tot)
        ## alle werte sind identisch, one layer 

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

        # calculate alpha for every head and cut with respective cutoff fn
        alpha_ij = self.vec_calc(self.coeff_fn, x_heads, w_heads, idx_i, idx_j)
        alpha_r_ij = (alpha_ij * phi_r_cut) #.permute(1,0).contiguous()
        alpha_s_ij = (alpha_ij * phi_chi_cut[:,None]) #.permute(1,0).contiguous()

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
    
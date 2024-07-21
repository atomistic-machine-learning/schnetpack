from typing import Callable, Dict, Union, Tuple, Sequence, Optional, Any
from functools import partial
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import schnetpack.properties as structure
from schnetpack.nn import ElectronicEmbedding
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.ops.spherical import order_contraction, make_l0_contraction_fn, interaction_order_contraction
from schnetpack.nn.utils import equal_head_split#, inv_split


import schnetpack.nn as snn
from schnetpack.nn.cutoff import ZeroCutoff

__all__ = [
    'So3kratesInteractionBlock',
    'So3kratesFeatureBlock',
    'So3kratesGeometricBlock',
    'So3kratesLayer',
    'So3krates'
    ]
        


class So3kratesInteractionBlock(nn.Module):

    """An interaction block consists of 
        - performs the cross degree coupling between the sphc vectors
        - coupling between atomic features and the sphc vectors
    """

    def __init__(
            self,
            num_features: int,
            degrees: Sequence[int],
            parity: bool = True
            ):
        
        super().__init__()
        self.register_buffer("degrees", torch.LongTensor(degrees))
        self.register_buffer("repeats", torch.LongTensor([2 * y + 1 for y in list(self.degrees)]))

        # whether to preserve parity of spherical features --> per default not considered and not used
        self.parity = parity
        num_segments = len(self.degrees)

        self.mixing_layer = snn.Dense(
            in_features=num_features+num_segments,
            out_features=num_features+num_segments,bias=False)


    def forward(self, x:torch.Tensor, chi:torch.Tensor):

        """
            Args:
                x: features coming from feature block shape: (n,F)
                chi: chi features coming from geometric block shape: (n,m_tot)
            Returns:
                b1_ cross-degree coupled chi features
                a1_ information updated atomic features from chi
        """
        F_ = x.shape[-1]
        nl = len(self.degrees)
        split_sizes = [F_,nl]

        # contract from (n_atoms, 2l+1) to (n_atoms, |l|)
        d_chi = interaction_order_contraction(chi,self.degrees)
        y = torch.concatenate([x, d_chi], axis=-1)  # shape: (n,F+|l|)

        # repeat first degree m_total order times (e.g for l0 its 3, and for l1 its 5) to sum up to total m_tot
        a1,b1 = torch.split(self.mixing_layer(y), split_sizes,dim=-1) # shape: (n,F) / shape: (n,n_l)
        b1_ = torch.repeat_interleave(input = b1,repeats=self.repeats,dim=-1) * chi
        return a1,b1_

class So3kratesFeatureBlock(nn.Module):

    """A feature block consists of 
        - a filter function combining
            the radial basis expanded atomic distances
            and the stacked per degree sphc, expanded in a norm or basis
        - an attention block for the message passing update step not using the SPHC
    """

    def __init__(self,filter_fn: Union[Callable, nn.Module],attention_fn: Union[Callable, nn.Module]):
        super().__init__()
        self.filter_fn = filter_fn
        self.attention_fn = attention_fn

    def forward(
            self,
            rbf:torch.Tensor, 
            d_gamma:torch.Tensor,
            x:torch.Tensor,
            idx_i:torch.Tensor,
            idx_j:torch.Tensor,
            phi_r_cut: torch.Tensor) -> torch.Tensor:

        w_ij = self.filter_fn(rbf,d_gamma) # shape: (n_pairs,F)
        x_ = self.attention_fn(
                            x=x,
                            w_ij=w_ij,
                            idx_j=idx_j,
                            idx_i=idx_i,
                            phi_r_cut=phi_r_cut) # shape (n,F)

        return x_

class So3kratesGeometricBlock(nn.Module):

    """A geometric block consists of 
        - a filter function combining
            the radial basis expanded atomic distances
            and the stacked per degree sphc, expanded in a norm or basis
        - an attention block for the message passing update step using the SPHC
    """

    def __init__(self,filter_fn: Union[Callable, nn.Module],attention_fn: Union[Callable, nn.Module]):
        super().__init__()
        self.filter_fn = filter_fn
        self.attention_fn = attention_fn

    def forward(
            self,
            chi: torch.Tensor,
            sph_ij: torch.Tensor,
            x:torch.Tensor,
            rbf:torch.Tensor, 
            d_gamma:torch.Tensor,
            idx_i:torch.Tensor,
            idx_j:torch.Tensor,
            phi_chi_cut:torch.Tensor,
            phi_r_cut: torch.Tensor) -> torch.Tensor:
        
        """
        Args:
            chi: spherical coordinates for all orders l, shape: (n,m_tot)
            sph_ij: spherical harmonics for all orders l, shape: (n_all_pairs,n,m_tot)
            x: atomic embeddings, shape: (n,F)
            rbf: radial basis expansion of distances, shape: (n_pairs,K)
            d_gamma: pairwise distance between spherical coordinates, shape: (n_all_pairs,|L|)
            phi_r_cut: filter cutoff, shape: (n_pairs,L)
            phi_chi_cut: cutoff that scales filter values based on distance in Spherical space,
                shape: (n_all_pairs,|L|)
            idx_i: index centering atom, shape: (n_pairs)
            idx_j: index neighboring atom, shape: (n_pairs)
        
        """
        w_ij = self.filter_fn(rbf,d_gamma) # shape: (n_pairs,F)
        chi_ = self.attention_fn(
                        chi=chi,  
                        sph_ij=sph_ij,
                        x=x,
                        w_ij=w_ij,
                        idx_i=idx_i,
                        idx_j = idx_j,
                        phi_r_cut=phi_r_cut,
                        phi_chi_cut=phi_chi_cut)
        return chi_ # shape: (n,m_tot)

class So3kratesLayer(nn.Module):

    def __init__(
            self,
            degrees: Sequence[int],
            feature_block: Union[Callable, nn.Module],
            geometry_block: Union[Callable, nn.Module],
            interaction_block: Union[Callable, nn.Module], 
            residual_mlp: Union[Callable,torch.nn.Module] = nn.Identity(),
            chi_cut_fn_dynamic:Union[Callable,torch.nn.Module] = ZeroCutoff(),
            layer_normalization: Union[Callable,torch.nn.Module] = nn.Identity()
            ):
        """
        Args:
            feature_block (Callable): function to calculate local features
            geometry_block (Callable): function to calculate geometric features
            interaction_block (Callable): function to calculate interaction between local and geometric features
                when passing an interaction block, the class should be written that it takes both x and chi
            degrees (Sequence[int]): degrees of spherical harmonics
            chi_cut_fn_dynamic (Callable): cutoff function for spherical harmonics
                if no cutoff function is provided, use zero cutoff
            layer_normalization (Callable): layer normalization if none given, 
                identiy which allows more consistent structure
            residual_mlp (Callable): residual mlp function, if none given 
                identity which allows more consistent structure
        """

        super().__init__()
        self.record = {}
        self.feature_block = feature_block
        self.geometry_block = geometry_block
        self.interaction_block = interaction_block
        self.layer_normalization = layer_normalization
        self.residual_mlp = residual_mlp
        self.chi_cut_fn_dynamic = chi_cut_fn_dynamic

        self.register_buffer("degrees", torch.LongTensor(list(degrees)))
        #self.reset_parameters()

    def helper(self,data,level,device):
        """this is just a helper function for debugging purposes"""
        weights,bias = (data[level]["weights"].T,data[level]["bias"])

        if len(weights.shape) == 1:
            weights = weights[:,None]

        weights = torch.nn.Parameter(torch.tensor(weights))
        if bias is not None:
            bias = torch.nn.Parameter(torch.tensor(bias))
        return weights,bias

    def reset_parameters(self) -> None:
        """ this is used for debugging (comparing implementation with original implementation)"""
        # DEBUG ONLY init with weights and bias of original implementation
        import numpy as np
        data = np.load("params.npz",allow_pickle=True)
        device = self.degrees.device



        # ['FeatureBlock_0', 'GeometricBlock_0', 'InteractionBlock_0']
        feature_block = data['FeatureBlock_0'].item()
        geometry_block = data['GeometricBlock_0'].item()
        interaction_block = data['InteractionBlock_0'].item()

        # step by step starting with feature block
        levels = list(feature_block.keys())
        # radial
        radial_levels = levels[:2]
        for n,level in enumerate(radial_levels):
            weights, bias = self.helper(feature_block,level,device)
            self.feature_block.filter_fn.rad_filter_fn.rad_filter_fn[n].weight = weights
            if bias is not None:
                self.feature_block.filter_fn.rad_filter_fn.rad_filter_fn[n].bias = bias

        # spherical 
        radial_levels = levels[2:4]
        for n,level in enumerate(radial_levels):
            weights, bias = self.helper(feature_block,level,device)
            self.feature_block.filter_fn.sph_filter_fn.sph_filter_fn[n].weight = weights
            if bias is not None:
                self.feature_block.filter_fn.sph_filter_fn.sph_filter_fn[n].bias = bias
        
        # attention coeffs
        attention_levels = levels[4:6]
        weights_A, bias_A = self.helper(feature_block,attention_levels[0],device)
        weights_B, bias_B = self.helper(feature_block,attention_levels[1],device)

        # Ansatz
        # so sind alle layer richtig
        self.feature_block.attention_fn.coeff_fn[0].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,0])
        self.feature_block.attention_fn.coeff_fn[0].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,0])
        
        self.feature_block.attention_fn.coeff_fn[1].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,1])
        self.feature_block.attention_fn.coeff_fn[1].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,1])


        # attention aggregation
        attention_agg_levels = levels[6:8]
        for n,level in enumerate(attention_agg_levels):
            weights, bias = self.helper(feature_block,level,device)
            # Ansatz: first layer und second layer aggregation sieht aus wie ref wenn weights.T loaded
            self.feature_block.attention_fn.aggregate_fn[0].v_j_linear.weight = torch.nn.Parameter(weights[:,:,0])
            self.feature_block.attention_fn.aggregate_fn[1].v_j_linear.weight = torch.nn.Parameter(weights[:,:,1])


        levels = list(geometry_block.keys())
        # radial
        radial_levels = levels[:2]
        for n,level in enumerate(radial_levels):
            weights, bias = self.helper(geometry_block,level,device)
            self.geometry_block.filter_fn.rad_filter_fn.rad_filter_fn[n].weight = weights
            if bias is not None:
                self.geometry_block.filter_fn.rad_filter_fn.rad_filter_fn[n].bias = bias
        # spherical
        radial_levels = levels[2:4]
        for n,level in enumerate(radial_levels):
            weights, bias = self.helper(geometry_block,level,device)
            self.geometry_block.filter_fn.sph_filter_fn.sph_filter_fn[n].weight = weights
            if bias is not None:
                self.geometry_block.filter_fn.sph_filter_fn.sph_filter_fn[n].bias = bias
        
        
        # attention coeffs
        attention_levels = levels[4:6]
        weights_A, bias_A = self.helper(geometry_block,attention_levels[0],device)
        weights_B, bias_B = self.helper(geometry_block,attention_levels[1],device)

        # Ansatz
        # so sind alle layer richtig
        self.geometry_block.attention_fn.coeff_fn[0].q_i_layer.weight = torch.nn.Parameter(weights_A)
        self.geometry_block.attention_fn.coeff_fn[0].k_j_layer.weight = torch.nn.Parameter(weights_B)
        


        levels = list(interaction_block.keys())
        for n,level in enumerate(levels):
            weights, bias = self.helper(interaction_block,level,device)
            self.interaction_block.mixing_layer.weight = weights
            if bias is not None:
                self.interaction_block.mixing_layer.bias = bias



    def forward(
            self,
            sph_ij: torch.Tensor, # shape: (n_pairs,m_tot) spherical harmonics from i to j
            chi: torch.Tensor, # shape: (n_pairs,m_tot) spherical harmonic coordinates init
            idx_j: torch.Tensor, # shape: (n_pairs,) idx of neighboring atom j
            idx_i: torch.Tensor, # shape: (n_pairs,) idx of centering atom i
            x: torch.Tensor,# shape: (n_atoms, F)  atomic features --> eco so aufbauen dass nonlocal features hierzu aufaddiert werden ?
            rbf: torch.Tensor, # shape: (n_pairs,K): rbf expanded distances
            phi_r_cut: torch.Tensor) -> torch.Tensor:
        
        # create m_tot contracted chi_ij
        self.record["chi_in"] = chi
        m_chi_ij = order_contraction(chi,idx_j,idx_i,self.degrees) # shape: (n_pairs, |l|)
        # apply pre layer normalization (for conv layer it may destroys spatial dependency)
        x_pre_1 = self.layer_normalization(x)
        # calculate phi_chi_cut
        phi_chi_cut = self.chi_cut_fn_dynamic(m_chi_ij)#[:,None] # TODO make sure that shape is consistent (npairs,1)

        # calculate local features
        x_local = self.feature_block(
                            rbf = rbf,
                            d_gamma = m_chi_ij,
                            x = x_pre_1,
                            idx_i = idx_i,
                            idx_j = idx_j,
                            phi_r_cut = phi_r_cut)

        # calculate geometric features
        chi_local = self.geometry_block(
                            chi = chi,
                            sph_ij = sph_ij,
                            x = x_pre_1,
                            rbf = rbf,
                            d_gamma = m_chi_ij,
                            phi_r_cut = phi_r_cut,
                            phi_chi_cut = phi_chi_cut,
                            idx_i = idx_i,
                            idx_j = idx_j,)

        # add local and sphc features, and first skip connection
        # different from original implementation, here it is assumed that nonlocal features
        # like charge or spin embedding are already added to input features x
        x_skip_1 = x + x_local
        chi_skip_1 = chi + chi_local

        # apply first residual mlp
        x_skip_1 = self.residual_mlp(x_skip_1)

        # apply second pre layer normalization
        x_pre_2 = self.layer_normalization(x_skip_1)

        # apply feature <-> sphc interaction layer
        delta_x, delta_chi = self.interaction_block(x_pre_2,chi_skip_1)

        # add second skip connection
        x_skip_2 = x_skip_1 + delta_x
        chi_skip_2 = chi_skip_1 + delta_chi

        # apply second residual mlp
        x_skip_2 = self.residual_mlp(x_skip_2)

        # apply final layer normalization
        x_skip_2 = self.layer_normalization(x_skip_2)

        # track chi results for later analysis
        self.record["chi_out"] = chi_skip_2

        # return final atomic features
        return x_skip_2, chi_skip_2
        


class So3krates(nn.Module):
    """So3krates implementation
    This is the from JAX ported implementation of So3krates  [#so3krates1].

    References:

    .. [#so3krates1] J.Thorben Frank, Oliver.T.Unke, Klaus.R Müller:
       So3krates: Equivariant attention for interactions on
       arbitrary length-scales in molecular systems.
       36th Conference NeurIPS, 2022.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_z: int = 101,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        activate_charge_spin_embedding: bool = False,
        embedding: Union[Callable, nn.Module] = None,
        degrees: Sequence[int] = [0,1,2],
        spherical_harmonics: nn.Module = None,
        so3krates_feature_block: nn.Module = None,
        so3krates_geometry_block: nn.Module = None,
        so3krates_interaction_block: nn.Module = None,
        so3krates_residual_mlp: nn.Module = None,
        so3krates_chi_cut_fn_dynamic: nn.Module = None,
        so3krates_layer_normalization: nn.Module = None,
    ):
        """
        TODO update args
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks, aka layers to be used in so3krates
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            max_z: maximal nuclear charge
            activation: activation function
            activate_charge_spin_embedding: if True, charge and spin embeddings are added to nuclear embeddings taken from SpookyNet Implementation
            embedding: type of nuclear embedding to use (simple is simple embedding and complex is the one with electron configuration)
            degrees: degrees of spherical harmonics
                in increasing order, and not skipping an order e.g [0,1,2] but not [0,1,3]
            spherical_harmonics: initial embedding for spherical harmonics distances
                probably containing redundant code (already present in so3) but for now keep it
            so3krates_feature_block: feature block for so3krates, for more details see So3kratesFeatureBlock
            so3krates_geometry_block: geometry block for so3krates, for more details see So3kratesGeometricBlock
            so3krates_interaction_block: interaction block for so3krates, for more details see So3kratesInteractionBlock
            so3krates_residual_mlp: residual mlp for so3krates, if none given identity which allows more consistent structure
            so3krates_chi_cut_fn_dynamic: cutoff function for spherical harmonics distances
                if no cutoff function is provided, use zero cutoff
            so3krates_layer_normalization: layer normalization if none given
                identiy which allows more consistent structure    
        """
        
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.activate_charge_spin_embedding = activate_charge_spin_embedding
        self.degrees = list(degrees)

        # initialize nuclear embedding
        self.embedding = embedding
        if self.embedding is None:

            # apply kaiming mormal initialization (He initialization), for better learning start
            nuc_weight_init = torch.empty((max_z,self.n_atom_basis))
            torch.nn.init.kaiming_normal_(nuc_weight_init, a=0, mode='fan_in', nonlinearity='linear')

            self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0,_weight=nuc_weight_init)

        # initialize spin and charge embeddings
        if self.activate_charge_spin_embedding:
            self.charge_embedding = ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=True)
            self.spin_embedding = ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=False)


        # spherical harmonics distances initial embedding
        self.spherical_harmonics = spherical_harmonics
        # initialize interaction blocks
        self.so3krates_layer = snn.replicate_module(
            lambda: So3kratesLayer(
                degrees=self.degrees,
                feature_block=so3krates_feature_block,
                geometry_block=so3krates_geometry_block,
                interaction_block=so3krates_interaction_block,
                residual_mlp=so3krates_residual_mlp,
                chi_cut_fn_dynamic=so3krates_chi_cut_fn_dynamic,
                layer_normalization=so3krates_layer_normalization
            ),
            self.n_interactions,
            False,
        )

    def helper(self,data,level,device):
        """this is just a helper function for debugging purposes"""
        weights,bias = (data[level]["weights"].T,data[level]["bias"])

        if len(weights.shape) == 1:
            weights = weights[:,None]

        weights = torch.nn.Parameter(torch.tensor(weights))
        if bias is not None:
            bias = torch.nn.Parameter(torch.tensor(bias))
        return weights,bias

    def reset_parameters(self) -> None:
        """ this is used for debugging (comparing implementation with original implementation)"""
        # DEBUG ONLY init with weights and bias of original implementation
        import numpy as np
        data = {}
        data = np.load("params_1.npz",allow_pickle=True)
        device = self.cutoff_fn.cutoff.device

        for M,layer in enumerate(list(data.keys())):

            # ['FeatureBlock_0', 'GeometricBlock_0', 'InteractionBlock_0']

            feature_block = data[layer].item()['FeatureBlock_0'].item()
            geometry_block = data[layer].item()['GeometricBlock_0'].item()
            interaction_block = data[layer].item()['InteractionBlock_0'].item()

            # step by step starting with feature block
            levels = list(feature_block.keys())
            # radial
            radial_levels = levels[:2]
            for n,level in enumerate(radial_levels):
                weights, bias = self.helper(feature_block,level,device)
                self.so3krates_layer[M].feature_block.filter_fn.rad_filter_fn.rad_filter_fn[n].weight = weights
                if bias is not None:
                    self.so3krates_layer[M].feature_block.filter_fn.rad_filter_fn.rad_filter_fn[n].bias = bias

            # spherical 
            radial_levels = levels[2:4]
            for n,level in enumerate(radial_levels):
                weights, bias = self.helper(feature_block,level,device)
                self.so3krates_layer[M].feature_block.filter_fn.sph_filter_fn.sph_filter_fn[n].weight = weights
                if bias is not None:
                    self.so3krates_layer[M].feature_block.filter_fn.sph_filter_fn.sph_filter_fn[n].bias = bias
            
            # attention coeffs
            attention_levels = levels[4:6]
            weights_A, bias_A = self.helper(feature_block,attention_levels[0],device)
            weights_B, bias_B = self.helper(feature_block,attention_levels[1],device)

            # Ansatz
            # so sind alle layer richtig
            self.so3krates_layer[M].feature_block.attention_fn.coeff_fn[0].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,0])
            self.so3krates_layer[M].feature_block.attention_fn.coeff_fn[0].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,0])
            
            self.so3krates_layer[M].feature_block.attention_fn.coeff_fn[1].q_i_layer.weight = torch.nn.Parameter(weights_A[:,:,1])
            self.so3krates_layer[M].feature_block.attention_fn.coeff_fn[1].k_j_layer.weight = torch.nn.Parameter(weights_B[:,:,1])


            # attention aggregation
            attention_agg_levels = levels[6:8]
            for n,level in enumerate(attention_agg_levels):
                weights, bias = self.helper(feature_block,level,device)
                # Ansatz: first layer und second layer aggregation sieht aus wie ref wenn weights.T loaded
                self.so3krates_layer[M].feature_block.attention_fn.aggregate_fn[0].v_j_linear.weight = torch.nn.Parameter(weights[:,:,0])
                self.so3krates_layer[M].feature_block.attention_fn.aggregate_fn[1].v_j_linear.weight = torch.nn.Parameter(weights[:,:,1])


            levels = list(geometry_block.keys())
            # radial
            radial_levels = levels[:2]
            for n,level in enumerate(radial_levels):
                weights, bias = self.helper(geometry_block,level,device)
                self.so3krates_layer[M].geometry_block.filter_fn.rad_filter_fn.rad_filter_fn[n].weight = weights
                if bias is not None:
                    self.so3krates_layer[M].geometry_block.filter_fn.rad_filter_fn.rad_filter_fn[n].bias = bias
            # spherical
            radial_levels = levels[2:4]
            for n,level in enumerate(radial_levels):
                weights, bias = self.helper(geometry_block,level,device)
                self.so3krates_layer[M].geometry_block.filter_fn.sph_filter_fn.sph_filter_fn[n].weight = weights
                if bias is not None:
                    self.so3krates_layer[M].geometry_block.filter_fn.sph_filter_fn.sph_filter_fn[n].bias = bias
            
            
            # attention coeffs
            attention_levels = levels[4:6]
            weights_A, bias_A = self.helper(geometry_block,attention_levels[0],device)
            weights_B, bias_B = self.helper(geometry_block,attention_levels[1],device)

            # Ansatz
            # so sind alle layer richtig
            self.so3krates_layer[M].geometry_block.attention_fn.coeff_fn[0].q_i_layer.weight = torch.nn.Parameter(weights_A)
            self.so3krates_layer[M].geometry_block.attention_fn.coeff_fn[0].k_j_layer.weight = torch.nn.Parameter(weights_B)
            


            levels = list(interaction_block.keys())
            for n,level in enumerate(levels):
                weights, bias = self.helper(interaction_block,level,device)
                self.so3krates_layer[M].interaction_block.mixing_layer.weight = weights
                if bias is not None:
                    self.so3krates_layer[M].interaction_block.mixing_layer.bias = bias

        print("DEBUG weights init done")


    def forward(self, inputs: Dict[str, torch.Tensor]):

        # get tensors from input dictionary
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i,idx_j = inputs[structure.idx_i],inputs[structure.idx_j]
        # compute pair features
        d_ij = torch.norm(r_ij, dim=1,keepdim=True)
        f_ij = self.radial_basis(d_ij.squeeze())
        rcut_ij = self.cutoff_fn(d_ij)

        # compute initial atomic embeddings
        x = self.embedding(atomic_numbers)

        # compute unit vectors
        unit_r_ij = r_ij / d_ij
        # compute initial spherical harmonics distances embeddings
        chi, g_ij,sph_ij = self.spherical_harmonics(unit_r_ij,rcut_ij,f_ij,atomic_numbers,idx_i)

        # add spin and charge embeddings
        if hasattr(self, "activate_charge_spin_embedding") and self.activate_charge_spin_embedding:
            # get tensors from input dictionary
            total_charge = inputs[structure.total_charge]
            spin = inputs[structure.spin_multiplicity]

            idx_m = inputs[structure.idx_m]
            num_batch = len(inputs[structure.idx])

            charge_embedding = self.charge_embedding(
                x, total_charge, num_batch, idx_m
            )
            spin_embedding = self.spin_embedding(
                x, spin, num_batch, idx_m
            )

            # additive combining of nuclear, charge and spin embedding
            x = x + charge_embedding + spin_embedding


        # compute interaction blocks and update atomic embeddings
        for so3krates_layer in self.so3krates_layer:
            
            v, chi_ = so3krates_layer(
                sph_ij=sph_ij,
                chi=chi,
                idx_j=idx_j,
                idx_i=idx_i,
                x=x,
                rbf=f_ij,
                phi_r_cut=rcut_ij
            )
            # the atomic embeddings are overwritten instead of adding the interaction
            # to the init embeddings, this is in contrast to schnet etc.
            x = v
            chi = chi_

        # collect results
        inputs["scalar_representation"] = x

        return inputs

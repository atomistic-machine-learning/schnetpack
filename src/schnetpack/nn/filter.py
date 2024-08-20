from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from functorch import combine_state_for_ensemble
import functorch
from schnetpack.nn import Dense

__all__ = [
    "RadialFilter",
    "SphericalFilter",
    "RadialSphericalFilter",
    "build_so3krates_mlp"
    ]
        


def build_so3krates_mlp(num_features: Sequence[int], activation: Callable) -> nn.Module:

    # TODO change to more general case
    # activation not for the last layer

    layers = []
    for n in range(len(num_features)-1):

        if n != len(num_features)-1:
            layer = Dense(
                        in_features=num_features[n], 
                        out_features=num_features[n+1],
                        activation=activation)
        else:
            layer = Dense(
                        in_features=num_features[n], 
                        out_features=num_features[n+1],
                        activation=None)
        layers.append(layer)

    return nn.ModuleList(layers)


class RadialFilter(nn.Module):

    def __init__(
            self,
            num_features: Sequence[int],
            activation: Callable = F.silu):
        
        super().__init__()
        self.num_features = num_features # [32,128]

        # filter network for the radial part
        # TODO should be adapted to work with arbitrary number of layers
        l1 = Dense(in_features=self.num_features[0],activation=activation,out_features=self.num_features[1])
        l2 = Dense(in_features=self.num_features[1],activation=None,out_features=self.num_features[1])

        self.rad_filter_fn = nn.Sequential(*[l1,l2])
        #self.rad_filter_fn = nn.ModuleList([l1,l2])

    def vec_calc(self, nn, x_heads, w_heads, idx_i, idx_j):
        '''to vectorize the head calculations, resulting in speed up'''
        fmodel, params = functorch.make_functional(nn)
        fmodel, params, buffers = combine_state_for_ensemble(nn)
        value =  functorch.vmap(
            fmodel, (0,0,0,0,None, None),out_dims=1)(
                params, 
                buffers, 
                x_heads,
                w_heads,
                idx_i,idx_j)
        return value


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        #alpha = self.vec_calc(self.coeff_fn, x_heads, w_heads, idx_i, idx_j)
        fmodel, params = functorch.make_functional(self.rad_filter_fn)
        x = fmodel(params, x)


        #for layer in self.rad_filter_fn:
        #    x = layer(x)
        return x


class SphericalFilter(nn.Module):

    def __init__(
        self,
        degrees: Sequence[int],
        num_features: int,
        activation: Callable = F.silu):
        
        super().__init__()
        self.register_buffer("degrees", torch.LongTensor(degrees))
        self.num_features = [len(self.degrees)] + list(num_features)  
    
        # TODO should be adapted to work with arbitrary number of layers
        l1 = Dense(in_features=16,activation=activation,out_features=self.num_features[1]) #set back to 32
        l2 = Dense(in_features=self.num_features[1],activation=None,out_features=self.num_features[2])
        # TODO check ob equivalent to sequential (sollte sein, aber sicherheitshalber checken)
        #self.sph_filter_fn = nn.ModuleList([l1,l2])
        self.sph_filter_fn  = nn.Sequential(*[l1,l2])

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        fmodel, params = functorch.make_functional(self.sph_filter_fn)
        x = fmodel(params, x)

        #for layer in self.sph_filter_fn:
        #    x = layer(x)
        return x


class RadialSphericalFilter(nn.Module):

    def __init__(
            self,
            num_features: Sequence[int],
            degrees: Sequence[int],
            activation: Callable = F.silu,
            debug_tag: str = None):
        
        super().__init__()
        self.register_buffer("degrees", torch.LongTensor(degrees))
        self.num_features =  list(num_features)
        # filter network for the radial part
        self.rad_filter_fn = RadialFilter(self.num_features,activation)
        # filter network for the spherical coordinates
        self.sph_filter_fn = SphericalFilter(self.degrees,num_features,activation)
        
    
    def forward(self,rbf:torch.Tensor,d_gamma:torch.Tensor) -> torch.Tensor:

        """
        Filter build from invariant geometric features,
        with K number of radial basis functions
        n_l number of degrees
        n_pairs number of pairwise distance pairs

        Args:
            rbf (Array): pairwise, radial basis expansion, shape: (n_pairs,K)
            d_gamma (Array): pairwise distance of spherical coordinates, shape: (n_pairs,n_l)
        """

        Wij = self.rad_filter_fn(rbf)
        Wij += self.sph_filter_fn(d_gamma)
        Wij = Wij.reshape(*rbf.shape[:-1],-1)
        return Wij



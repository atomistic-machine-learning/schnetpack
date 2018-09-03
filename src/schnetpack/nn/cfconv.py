import torch
from torch import nn as nn
from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate

__all__ = [
    'CFConv'
]


class CFConv(nn.Module):
    """
    Continuous-filter convolution layer for SchNet.

    Args:
        n_in (int): Number of input dimensions
        n_filters (int): Number of filter dimensions
        n_out (int): Number of output dimensions
        filter_network (nn.Module): Calculates filter
        cutoff_network (nn.Module): Calculates optional cutoff function (default: None)
        activation (function): Activation function
        normalize_filter (bool): If true, normalize filter to number of neighbors (default: false)
        axis (int): axis over which convolution should be applied
    """

    def __init__(self, n_in, n_filters, n_out, filter_network, cutoff_network=None,
                 activation=None, normalize_filter=False, axis=2):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False)
        self.f2out = Dense(n_filters, n_out, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Input representation of atomic environments.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            pairwise_mask (torch.Tensor): Mask to filter out non-existing neighbors introduced via padding.
            f_ij (torch.Tensor): Use at your own risk. Set to None otherwise.

        Returns:
            torch.Tensor: Continuous convolution.

        """
        if f_ij is None:
            f_ij = r_ij

        # calculate filter
        W = self.filter_network(f_ij)

        # apply optional cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            #            print(C)
            W *= C

        # convolution
        y = self.in2f(x)

        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))

        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = y * W
        y = self.agg(y, pairwise_mask)

        y = self.f2out(y)

        return y

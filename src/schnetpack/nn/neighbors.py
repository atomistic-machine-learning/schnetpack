import torch
from torch import nn as nn


class AtomDistances(nn.Module):
    """
    Layer that calculates all pair-wise distances between atoms.

    Use advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors. Indices of the
    neighbors to consider are stored in neighbors.
    """

    def __init__(self):
        super(AtomDistances, self).__init__()

    def forward(self, positions, neighbors, cell=None, cell_offsets=None):
        """
        Args:
            positions (torch.Tensor): Atomic positions, differentiable torch Variable (B x N_at x 3)
            neighbors (torch.Tensor): Indices of neighboring atoms (B x N_at x N_nbh)
            cell (torch.tensor): cell for periodic systems (B x 3 x 3)
            cell_offsets (torch.Tensor): offset of atom in cell coordinates (B x N_at x N_nbh x 3)

        Returns:
            torch.Tensor: Distances of every atoms to its neighbors (B x N_at x N_nbh)
        """
        return atom_distances(positions, neighbors, cell, cell_offsets)


def triple_distances(positions, neighbors_j, neighbors_k):
    """
    Get all distances between atoms forming a triangle with the central atoms.
    Required e.g. for angular symmetry functions.

    Args:
        positions (torch.Tensor): Atomic positions
        neighbors_j (torch.Tensor): Indices of first neighbor in triangle
        neighbors_k (torch.Tensor): Indices of second neighbor in triangle

    Returns:
        torch.Tensor: Distance between central atom and neighbor j
        torch.Tensor: Distance between central atom and neighbor k
        torch.Tensor: Distance between neighbors

    """
    nbatch, natoms, nangles = neighbors_k.size()
    idx_m = torch.arange(nbatch, device=positions.device, dtype=torch.long)[:, None, None]
    # if positions.is_cuda:
    #    idx_m = idx_m.pin_memory().cuda(async=True)
    pos_j = positions[idx_m, neighbors_j[:], :]
    pos_k = positions[idx_m, neighbors_k[:], :]
    R_ij = pos_j - positions[:, :, None, :]
    R_ik = pos_k - positions[:, :, None, :]
    R_jk = pos_j - pos_k
    r_ij = torch.norm(R_ij, 2, 3)
    r_ik = torch.norm(R_ik, 2, 3)
    r_jk = torch.norm(R_jk, 2, 3)
    return r_ij, r_ik, r_jk


class TriplesDistances(nn.Module):
    """
    Layer that gets all distances between atoms forming a triangle with the central atoms.
    Required e.g. for angular symmetry functions.
    """

    def __init__(self):
        super(TriplesDistances, self).__init__()

    def forward(self, positions, neighbors_j, neighbors_k):
        """
        Args:
            positions (torch.Tensor): Atomic positions
            neighbors_j (torch.Tensor): Indices of first neighbor in triangle
            neighbors_k (torch.Tensor): Indices of second neighbor in triangle

        Returns:
            torch.Tensor: Distance between central atom and neighbor j
            torch.Tensor: Distance between central atom and neighbor k
            torch.Tensor: Distance between neighbors

        """
        return triple_distances(positions, neighbors_j, neighbors_k)


def neighbor_elements(atomic_numbers, neighbors):
    """
    Return the atomic numbers associated with the neighboring atoms. Can also be used to gather other properties by
    neighbors if different atom-wise Tensor is passed instead of atomic_numbers.

    Args:
        atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
        neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

    Returns:
        torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)

    """
    # Get molecules in batch
    n_batch = atomic_numbers.size()[0]
    # Construct auxiliary index
    idx_m = torch.arange(n_batch, device=atomic_numbers.device, dtype=torch.long)[:, None, None]
    # Get neighbors via advanced indexing
    neighbor_numbers = atomic_numbers[idx_m, neighbors[:, :, :]]
    return neighbor_numbers


class NeighborElements(nn.Module):
    """
    Layer to obtain the atomic numbers associated with the neighboring atoms.
    """

    def __init__(self):
        super(NeighborElements, self).__init__()

    def forward(self, atomic_numbers, neighbors):
        """
        Args:
            atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
            neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

        Returns:
            torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)
        """
        return neighbor_elements(atomic_numbers, neighbors)


def atom_distances(positions, neighbors, cell=None, cell_offsets=None):
    """
    Use advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors. Indices of the
    neighbors to consider are stored in neighbors.

    Args:
        positions (torch.Tensor): Atomic positions, differentiable torch Variable (B x N_at x 3)
        neighbors (torch.Tensor): Indices of neighboring atoms (B x N_at x N_nbh)
        cell (torch.Tensor): cell for periodic systems (B x 3 x 3)
        cell_offsets (torch.Tensor): offset of atom in cell coordinates (B x N_at x N_nbh x 3)

    Returns:
        torch.Tensor: Distances of every atom to its neighbors (B x N_at x N_nbh)
    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[:, None, None]
    # Get atomic positions of all neighboring indices
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    # Compute vector lengths
    distances = torch.norm(dist_vec, 2, 3)
    return distances

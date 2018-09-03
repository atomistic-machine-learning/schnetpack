import numpy as np
from ase.neighborlist import neighbor_list


class BaseEnvironmentProvider:
    """
    Environment Providers are supposed to collect neighboring atoms within local, atom-centered environments.
    All environment providers should inherit from this class.
    """

    def get_environment(self, idx, atoms):
        '''
        Returns the neighbor indices and offsets

        Args:
            idx (int): index in the data set
            atoms (ase.Atoms): atomistic system

        Returns:
            neighborhood_idx (np.ndarray): indices of the neighbors with shape n_atoms x n_max_neighbors
            offset (np.ndarray): offset in lattice coordinates for periodic systems (otherwise zero matrix) of
            shape n_atoms x n_max_neighbors x 3

        '''

        raise NotImplementedError


class SimpleEnvironmentProvider(BaseEnvironmentProvider):
    '''
    A simple environment provider for small molecules where all atoms are each other's neighbors.
    It calculates full distance matrices and does not support cutoffs or periodic boundary conditions.
    '''

    def get_environment(self, idx, atoms, grid=None):
        n_atoms = atoms.get_number_of_atoms()

        if n_atoms == 1:
            neighborhood_idx = -np.ones((1, 1), dtype=np.float32)
            offsets = np.zeros((n_atoms, 1, 3), dtype=np.float32)
        else:
            neighborhood_idx = np.tile(np.arange(n_atoms, dtype=np.float32)[np.newaxis], (n_atoms, 1))
            neighborhood_idx = neighborhood_idx[~np.eye(n_atoms, dtype=np.bool)].reshape(n_atoms, n_atoms - 1)

            if grid is not None:
                n_grid = grid.shape[0]
                neighborhood_idx = np.hstack([neighborhood_idx, -np.ones((n_atoms, 1))])
                grid_nbh = np.tile(np.arange(n_atoms, dtype=np.float32)[np.newaxis], (n_grid, 1))
                neighborhood_idx = np.vstack([neighborhood_idx, grid_nbh])

            offsets = np.zeros((neighborhood_idx.shape[0], neighborhood_idx.shape[1], 3), dtype=np.float32)
        return neighborhood_idx, offsets


class ASEEnvironmentProvider:
    '''
    Environment provider making use of ASE neighbor lists. Supports cutoffs and PBCs.
    '''

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def get_environment(self, idx, atoms, grid=None):
        if grid is not None:
            raise NotImplementedError

        n_atoms = atoms.get_number_of_atoms()
        idx_i, idx_j, idx_S = neighbor_list('ijS', atoms, self.cutoff, self_interaction=False)
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1))

            mask = np.zeros((n_atoms, np.max(n_max_nbh)), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((n_atoms, np.max(n_max_nbh)), dtype=np.float32)
            neighborhood_idx[mask] = idx_j

            offset = np.zeros((n_atoms, np.max(n_max_nbh), 3), dtype=np.float32)
            offset[mask] = idx_S
        else:
            neighborhood_idx = -np.ones((n_atoms, 1), dtype=np.float32)
            offset = np.zeros((n_atoms, 1, 3), dtype=np.float32)

        return neighborhood_idx, offset


def collect_atom_triples(nbh_idx):
    """
    Collect all valid triples of atoms by rearranging neighbor indices obtained from an environment provider.

    Args:
        nbh_idx (numpy.ndarray): neighbor indices

    Returns:
        nbh_idx_j, nbh_idx_k (numpy.ndarray): triple indices

    """
    natoms, nneigh = nbh_idx.shape
    # Construct possible permutations
    nbh_idx_j = np.tile(nbh_idx, nneigh)
    nbh_idx_k = np.repeat(nbh_idx, nneigh).reshape((natoms, -1))
    # Remove same interactions and non unique pairs
    triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
    triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
    nbh_idx_j = nbh_idx_j[:, triu_idx_flat]
    nbh_idx_k = nbh_idx_k[:, triu_idx_flat]
    return nbh_idx_j, nbh_idx_k

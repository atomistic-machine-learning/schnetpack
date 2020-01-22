import math

import numpy as np
import torch
from ase.neighborlist import neighbor_list

__all__ = [
    "BaseEnvironmentProvider",
    "SimpleEnvironmentProvider",
    "AseEnvironmentProvider",
    "TorchEnvironmentProvider",
]


class BaseEnvironmentProvider:
    """
    Environment Providers are supposed to collect neighboring atoms within
    local, atom-centered environments. All environment providers should inherit
    from this class.

    """

    def get_environment(self, atoms):
        """
        Returns the neighbor indices and offsets

        Args:
            atoms (ase.Atoms): atomistic system

        Returns:
            neighborhood_idx (np.ndarray): indices of the neighbors with shape
                n_atoms x n_max_neighbors
            offset (np.ndarray): offset in lattice coordinates for periodic
                systems (otherwise zero matrix) of shape
                n_atoms x n_max_neighbors x 3

        """

        raise NotImplementedError


class SimpleEnvironmentProvider(BaseEnvironmentProvider):
    """
    A simple environment provider for small molecules where all atoms are each
    other's neighbors. It calculates full distance matrices and does not
    support cutoffs or periodic boundary conditions.
    """

    def get_environment(self, atoms, grid=None):
        n_atoms = atoms.get_global_number_of_atoms()

        if n_atoms == 1:
            neighborhood_idx = -np.ones((1, 1), dtype=np.float32)
            offsets = np.zeros((n_atoms, 1, 3), dtype=np.float32)
        else:
            neighborhood_idx = np.tile(
                np.arange(n_atoms, dtype=np.float32)[np.newaxis], (n_atoms, 1)
            )

            neighborhood_idx = neighborhood_idx[
                ~np.eye(n_atoms, dtype=np.bool)
            ].reshape(n_atoms, n_atoms - 1)

            if grid is not None:
                n_grid = grid.shape[0]
                neighborhood_idx = np.hstack([neighborhood_idx, -np.ones((n_atoms, 1))])
                grid_nbh = np.tile(
                    np.arange(n_atoms, dtype=np.float32)[np.newaxis], (n_grid, 1)
                )
                neighborhood_idx = np.vstack([neighborhood_idx, grid_nbh])

            offsets = np.zeros(
                (neighborhood_idx.shape[0], neighborhood_idx.shape[1], 3),
                dtype=np.float32,
            )
        return neighborhood_idx, offsets


class AseEnvironmentProvider(BaseEnvironmentProvider):
    """
    Environment provider making use of ASE neighbor lists. Supports cutoffs
    and PBCs.

    """

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def get_environment(self, atoms, grid=None):
        if grid is not None:
            raise NotImplementedError

        n_atoms = atoms.get_global_number_of_atoms()
        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", atoms, self.cutoff, self_interaction=False
        )
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

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


class TorchEnvironmentProvider(BaseEnvironmentProvider):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).
    Supports cutoffs, PBCs and can be performed on either CPU or GPU.

    """

    def __init__(self, cutoff, device):
        """
        Args:
            cutoff (float): the cutoff inside which atoms are considered pairs
            device (:class:`torch.device`): pass torch.device('cpu') or torch.device('cuda') to
                perform the calculation on a CPU or GPU, respectively.
        """
        self.cutoff = cutoff
        self.device = device

    def get_environment(self, atoms):

        species = torch.FloatTensor(atoms.numbers).to(self.device)
        coordinates = torch.FloatTensor(atoms.positions).to(self.device)
        pbc = torch.from_numpy(atoms.pbc.astype("uint8")).to(self.device)

        if not atoms.cell.any():
            cell = torch.eye(3, dtype=species.dtype).to(self.device)
        else:
            cell = torch.Tensor(atoms.cell).to(self.device)

        shifts = compute_shifts(cell=cell, pbc=pbc, cutoff=self.cutoff)

        # The returned indices are only one directional
        idx_i, idx_j, idx_S = neighbor_pairs(
            species == -1, coordinates, cell, shifts, self.cutoff
        )

        idx_i = idx_i.cpu().detach().numpy()
        idx_j = idx_j.cpu().detach().numpy()
        idx_S = idx_S.cpu().detach().numpy()

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = np.hstack((idx_i, idx_j))
        bi_idx_j = np.hstack((idx_j, idx_i))
        bi_idx_S = np.vstack((-idx_S, idx_S))

        n_atoms = atoms.get_global_number_of_atoms()
        if bi_idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(bi_idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

            mask = np.zeros((n_atoms, np.max(n_max_nbh)), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((n_atoms, np.max(n_max_nbh)), dtype=np.float32)
            offset = np.zeros((n_atoms, np.max(n_max_nbh), 3), dtype=np.float32)

            # Assign neighbors and offsets according to the indices in bi_idx_i, since in contrast
            # to the ASE provider the bidirectional arrays are no longer sorted.
            # TODO: There might be a more efficient way of doing this than a loop
            for idx in range(n_atoms):
                neighborhood_idx[idx, mask[idx]] = bi_idx_j[bi_idx_i == idx]
                offset[idx, mask[idx]] = bi_idx_S[bi_idx_i == idx]

        else:
            neighborhood_idx = -np.ones((n_atoms, 1), dtype=np.float32)
            offset = np.zeros((n_atoms, 1, 3), dtype=np.float32)

        return neighborhood_idx, offset


def compute_shifts(cell, pbc, cutoff):
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration.
    Copyright 2018- Xiang Gao and other ANI developers
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
        vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: long tensor of shifts. the center cell and
            symmetric cells are not included.
    """
    # type: (Tensor, Tensor, float) -> Tensor
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))

    r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
    r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
    r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)

    return torch.cat(
        [
            torch.cartesian_prod(r1, r2, r3),
            torch.cartesian_prod(r1, r2, o),
            torch.cartesian_prod(r1, r2, -r3),
            torch.cartesian_prod(r1, o, r3),
            torch.cartesian_prod(r1, o, o),
            torch.cartesian_prod(r1, o, -r3),
            torch.cartesian_prod(r1, -r2, r3),
            torch.cartesian_prod(r1, -r2, o),
            torch.cartesian_prod(r1, -r2, -r3),
            torch.cartesian_prod(o, r2, r3),
            torch.cartesian_prod(o, r2, o),
            torch.cartesian_prod(o, r2, -r3),
            torch.cartesian_prod(o, o, r3),
        ]
    )


def neighbor_pairs(padding_mask, coordinates, cell, shifts, cutoff):
    """Compute pairs of atoms that are neighbors
    Copyright 2018- Xiang Gao and other ANI developers
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
            defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
    """
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    coordinates = coordinates.detach()
    cell = cell.detach()
    num_atoms = padding_mask.shape[0]
    all_atoms = torch.arange(num_atoms, device=cell.device)

    # Step 2: center cell
    p1_center, p2_center = torch.combinations(all_atoms).unbind(-1)
    shifts_center = shifts.new_zeros(p1_center.shape[0], 3)

    # Step 3: cells with shifts
    # shape convention (shift index, molecule index, atom index, 3)
    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    shift_index, p1, p2 = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).unbind(
        -1
    )
    shifts_outside = shifts.index_select(0, shift_index)

    # Step 4: combine results for all cells
    shifts_all = torch.cat([shifts_center, shifts_outside])
    p1_all = torch.cat([p1_center, p1])
    p2_all = torch.cat([p2_center, p2])

    shift_values = torch.mm(shifts_all.to(cell.dtype), cell)

    # step 5, compute distances, and find all pairs within cutoff
    distances = (coordinates[p1_all] - coordinates[p2_all] + shift_values).norm(2, -1)

    padding_mask = (padding_mask[p1_all]) | (padding_mask[p2_all])
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = (distances < cutoff).nonzero()
    pair_index = in_cutoff.squeeze()
    atom_index1 = p1_all[pair_index]
    atom_index2 = p2_all[pair_index]
    shifts = shifts_all.index_select(0, pair_index)
    return atom_index1, atom_index2, shifts


def collect_atom_triples(nbh_idx):
    """
    Collect all valid triples of atoms by rearranging neighbor indices obtained
    from an environment provider.

    Args:
        nbh_idx (numpy.ndarray): neighbor indices

    Returns:
        nbh_idx_j, nbh_idx_k (numpy.ndarray): triple indices
        offset_idx_j, offset_idx_k (numpy.ndarray): offset indices for PBC

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

    # Keep track of periodic images
    offset_idx = np.tile(np.arange(nneigh), (natoms, 1))

    # Construct indices for pairs of offsets
    offset_idx_j = np.tile(offset_idx, nneigh)
    offset_idx_k = np.repeat(offset_idx, nneigh).reshape((natoms, -1))

    # Remove non-unique pairs and diagonal
    offset_idx_j = offset_idx_j[:, triu_idx_flat]
    offset_idx_k = offset_idx_k[:, triu_idx_flat]

    return nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k

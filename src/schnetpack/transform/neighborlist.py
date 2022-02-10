import os
import torch
import shutil
from ase import Atoms
from ase.neighborlist import neighbor_list
from typing import Dict
from .base import Transform
from dirsync import sync

__all__ = [
    "ASENeighborList",
    "TorchNeighborList",
    "CountNeighbors",
    "CollectAtomTriples",
    "CachedNeighborList",
    "NeighborListTransform",
    "WrapPositions",
]

from schnetpack import properties
import fasteners


class CacheException(Exception):
    pass


class CachedNeighborList(Transform):
    """
    Dynamic caching of neighbor lists.
    This wraps a neighbor list and stores the results the first time it is called
    for a dataset entry with the pid provided by AtomsDataset. Particularly,
    for large systems, this speeds up training significantly.
    Note:
        The provided cache location should be unique to the used dataset. Otherwise,
        wrong neighborhoods will be provided. The caching location can be reused
        across multiple runs, by setting `cleanup_cache=False`.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        cache_path: str,
        neighbor_list: Transform,
        keep_cache: bool = False,
        cache_workdir: str = None,
    ):
        """
        Args:
            cache_path: Path of caching directory.
            neighbor_list: the neighbor list to use
            keep_cache: Keep cache at `cache_location` at the end of training, or copy built/updated cache there from
                `cache_workdir` (if set). A pre-existing cache at `cache_location` will not be deleted, while a
                 temporary cache at `cache_workdir` will always be removed.
            cache_workdir: If this is set, the cache will be build here, e.g. a cluster scratch space
                for faster performance. An existing cache at `cache_location` is copied here at the beginning of
                training, and afterwards (if `keep_cache=True`) the final cache is copied to `cache_workdir`.
        """
        super().__init__()
        self.neighbor_list = neighbor_list
        self.keep_cache = keep_cache
        self.cache_path = cache_path
        self.cache_workdir = cache_workdir
        self.preexisting_cache = os.path.exists(self.cache_path)
        self.has_tmp_workdir = cache_workdir is not None

        os.makedirs(cache_path, exist_ok=True)

        if self.has_tmp_workdir:
            # cache workdir should be empty to avoid loading nbh lists from earlier runs
            if os.path.exists(cache_workdir):
                raise CacheException("The provided `cache_workdir` already exists!")

            # copy existing nbh lists to cache workdir
            if self.preexisting_cache:
                shutil.copytree(cache_path, cache_workdir)
            self.cache_location = cache_workdir
        else:
            # use cache_location to store and load neighborlists
            self.cache_location = cache_path

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cache_file = os.path.join(
            self.cache_location, f"cache_{inputs[properties.idx][0]}.pt"
        )

        # try to read cached NBL
        try:
            data = torch.load(cache_file)
            inputs.update(data)
        except IOError:
            # acquire lock for caching
            lock = fasteners.InterProcessLock(
                os.path.join(
                    self.cache_location, f"cache_{inputs[properties.idx][0]}.lock"
                )
            )
            with lock:
                # retry reading, in case other process finished in the meantime
                try:
                    data = torch.load(cache_file)
                    inputs.update(data)
                except IOError:
                    # now it is save to calculate and cache
                    inputs = self.neighbor_list(inputs)
                    data = {
                        properties.idx_i: inputs[properties.idx_i],
                        properties.idx_j: inputs[properties.idx_j],
                        properties.offsets: inputs[properties.offsets],
                    }
                    torch.save(data, cache_file)
                except Exception as e:
                    print(e)
        return inputs

    def teardown(self):
        if not self.keep_cache and not self.preexisting_cache:
            try:
                shutil.rmtree(self.cache_path)
            except:
                pass

        if self.cache_workdir is not None:
            if self.keep_cache:
                try:
                    sync(self.cache_workdir, self.cache_path, "sync")
                except:
                    pass

            try:
                shutil.rmtree(self.cache_workdir)
            except:
                pass


class NeighborListTransform(Transform):
    """
    Base class for neighbor lists.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        cutoff: float,
    ):
        """
        Args:
            cutoff: Cutoff radius for neighbor search.
        """
        super().__init__()
        self._cutoff = cutoff

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        Z = inputs[properties.Z]
        R = inputs[properties.R]
        cell = inputs[properties.cell].view(3, 3)
        pbc = inputs[properties.pbc]

        idx_i, idx_j, offset = self._build_neighbor_list(Z, R, cell, pbc, self._cutoff)
        inputs[properties.idx_i] = idx_i.detach()
        inputs[properties.idx_j] = idx_j.detach()
        inputs[properties.offsets] = offset
        return inputs

    def _build_neighbor_list(
        self,
        Z: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
    ):
        """Override with specific neighbor list implementation"""
        raise NotImplementedError


class ASENeighborList(NeighborListTransform):
    """
    Calculate neighbor list using ASE.
    """

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        at = Atoms(numbers=Z, positions=positions, cell=cell, pbc=pbc)

        idx_i, idx_j, S = neighbor_list("ijS", at, cutoff, self_interaction=False)
        idx_i = torch.from_numpy(idx_i)
        idx_j = torch.from_numpy(idx_j)
        S = torch.from_numpy(S).to(dtype=positions.dtype)
        offset = torch.mm(S, cell)
        return idx_i, idx_j, offset


class TorchNeighborList(NeighborListTransform):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).
    Supports cutoffs and PBCs and can be performed on either CPU or GPU.
    """

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device, dtype=torch.long)
        else:
            shifts = self._get_shifts(cell, pbc, cutoff)
        idx_i, idx_j, offset = self._get_neighbor_pairs(positions, cell, shifts, cutoff)

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((idx_i, idx_j), dim=0)
        bi_idx_j = torch.cat((idx_j, idx_i), dim=0)

        # Sort along first dimension (necessary for atom-wise pooling)
        sorted_idx = torch.argsort(bi_idx_i)
        idx_i = bi_idx_i[sorted_idx]
        idx_j = bi_idx_j[sorted_idx]

        bi_offset = torch.cat((-offset, offset), dim=0)
        offset = bi_offset[sorted_idx]
        offset = torch.mm(offset.to(cell.dtype), cell)

        return idx_i, idx_j, offset

    def _get_neighbor_pairs(self, positions, cell, shifts, cutoff):
        """Compute pairs of atoms that are neighbors
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # 1) Central cell
        pi_center, pj_center = torch.combinations(all_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, all_atoms, all_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoff
        distances = torch.norm(Rij_all, dim=1)
        in_cutoff = torch.nonzero(distances < cutoff, as_tuple=False)

        # 6) Reduce tensors to relevant components
        pair_index = in_cutoff.squeeze()
        atom_index_i = pi_all[pair_index]
        atom_index_j = pj_all[pair_index]
        offsets = shifts_all[pair_index]

        return atom_index_i, atom_index_j, offsets

    def _get_shifts(self, cell, pbc, cutoff):
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.
        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(cutoff * inverse_lengths).long()
        num_repeats = torch.where(
            pbc, num_repeats, torch.Tensor([0], device=cell.device).long()
        )

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


class CollectAtomTriples(Transform):
    """
    Generate the index tensors for all triples between atoms within the cutoff shell.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Using the neighbors contained within the cutoff shell, generate all unique pairs of neighbors and convert
        them to index arrays. Applied to the neighbor arrays, these arrays generate the indices involved in the atom
        triples.
        E.g.:
            idx_j[idx_j_triples] -> j atom in triple
            idx_j[idx_k_triples] -> k atom in triple
            Rij[idx_j_triples] -> Rij vector in triple
            Rij[idx_k_triples] -> Rik vector in triple
        """
        idx_i = inputs[properties.idx_i]

        _, n_neighbors = torch.unique_consecutive(idx_i, return_counts=True)

        offset = 0
        idx_i_triples = ()
        idx_jk_triples = ()
        for idx in range(n_neighbors.shape[0]):
            triples = torch.combinations(
                torch.arange(offset, offset + n_neighbors[idx]), r=2
            )
            idx_i_triples += (torch.ones(triples.shape[0], dtype=torch.long) * idx,)
            idx_jk_triples += (triples,)
            offset += n_neighbors[idx]

        idx_i_triples = torch.cat(idx_i_triples)

        idx_jk_triples = torch.cat(idx_jk_triples)
        idx_j_triples, idx_k_triples = idx_jk_triples.split(1, dim=-1)

        inputs[properties.idx_i_triples] = idx_i_triples
        inputs[properties.idx_j_triples] = idx_j_triples.squeeze(-1)
        inputs[properties.idx_k_triples] = idx_k_triples.squeeze(-1)
        return inputs


class CountNeighbors(Transform):
    """
    Store the number of neighbors for each atom
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, sorted: bool = True):
        """
        Args:
            sorted: Set to false if chosen neighbor list yields unsorted center indices (idx_i).
        """
        super(CountNeighbors, self).__init__()
        self.sorted = sorted

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        idx_i = inputs[properties.idx_i]

        if self.sorted:
            _, n_nbh = torch.unique_consecutive(idx_i, return_counts=True)
        else:
            _, n_nbh = torch.unique(idx_i, return_counts=True)

        inputs[properties.n_nbh] = n_nbh
        return inputs


class WrapPositions(Transform):
    """
    Wrap atom positions into periodic cell. This routine requires a non-zero cell.
    The cell center of the inverse cell is set to (0.5, 0.5, 0.5).
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps (float): small offset for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        R = inputs[properties.R]
        cell = inputs[properties.cell].view(3, 3)
        pbc = inputs[properties.pbc]

        inverse_cell = torch.inverse(cell)
        inv_positions = torch.sum(R[..., None] * inverse_cell[None, ...], dim=1)

        periodic = torch.masked_select(inv_positions, pbc[None, ...])

        # Apply periodic boundary conditions (with small buffer)
        periodic = periodic + self.eps
        periodic = periodic % 1.0
        periodic = periodic - self.eps

        # Update fractional coordinates
        inv_positions.masked_scatter_(pbc[None, ...], periodic)

        # Convert to positions
        R_wrapped = torch.sum(inv_positions[..., None] * cell[None, ...], dim=1)

        inputs[properties.R] = R_wrapped

        return inputs

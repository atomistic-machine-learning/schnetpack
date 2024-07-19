import os
import torch
import shutil
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list
from matscipy.neighbours import neighbour_list as msp_neighbor_list
from .base import Transform
from dirsync import sync
import numpy as np
from typing import Optional, Dict, List

__all__ = [
    "ASENeighborList",
    "MatScipyNeighborList",
    "TorchNeighborList",
    "CountNeighbors",
    "CollectAtomTriples",
    "CachedNeighborList",
    "NeighborListTransform",
    "WrapPositions",
    "SkinNeighborList",
    "FilterNeighbors",
]

import schnetpack as spk
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
        across multiple runs, by setting `keep_cache=True`.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        cache_path: str,
        neighbor_list: Transform,
        nbh_transforms: Optional[List[torch.nn.Module]] = None,
        keep_cache: bool = False,
        cache_workdir: str = None,
    ):
        """
        Args:
            cache_path: Path of caching directory.
            neighbor_list: the neighbor list to use
            nbh_transforms: transforms for manipulating the neighbor lists
                provided by neighbor_list
            keep_cache: Keep cache at `cache_location` at the end of training, or copy
                built/updated cache there from `cache_workdir` (if set). A pre-existing
                cache at `cache_location` will not be deleted, while a temporary cache
                at `cache_workdir` will always be removed.
            cache_workdir: If this is set, the cache will be build here, e.g. a cluster
                scratch space for faster performance. An existing cache at
                `cache_location` is copied here at the beginning of training, and
                afterwards (if `keep_cache=True`) the final cache is copied to
                `cache_workdir`.
        """
        super().__init__()
        self.neighbor_list = neighbor_list
        self.nbh_transforms = nbh_transforms or []
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
                    for nbh_transform in self.nbh_transforms:
                        inputs = nbh_transform(inputs)
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

        idx_i, idx_j, S = ase_neighbor_list("ijS", at, cutoff, self_interaction=False)
        idx_i = torch.from_numpy(idx_i)
        idx_j = torch.from_numpy(idx_j)
        S = torch.from_numpy(S).to(dtype=positions.dtype)
        offset = torch.mm(S, cell)
        return idx_i, idx_j, offset


class MatScipyNeighborList(NeighborListTransform):
    """
    Neighborlist using the efficient implementation of the Matscipy package

    References:
        https://github.com/libAtoms/matscipy
    """

    def _build_neighbor_list(
        self, Z, positions, cell, pbc, cutoff, eps=1e-6, buffer=1.0
    ):
        at = Atoms(numbers=Z, positions=positions, cell=cell, pbc=pbc)

        # Add cell if none is present (volume = 0)
        if at.cell.volume < eps:
            # max values - min values along xyz augmented by small buffer for stability
            new_cell = np.ptp(at.positions, axis=0) + buffer
            # Set cell and center
            at.set_cell(new_cell, scale_atoms=False)
            at.center()

        # Compute neighborhood
        idx_i, idx_j, S = msp_neighbor_list("ijS", at, cutoff)
        idx_i = torch.from_numpy(idx_i).long()
        idx_j = torch.from_numpy(idx_j).long()
        S = torch.from_numpy(S).to(dtype=positions.dtype)
        offset = torch.mm(S, cell)

        return idx_i, idx_j, offset


class SkinNeighborList(Transform):
    """
    Neighbor list provider utilizing a cutoff skin for computational efficiency. Wrapper
    around neighbor list classes such as, e.g., ASENeighborList. Designed for use cases
    with gradual structural changes such ase MD simulations and structure relaxations.

    Note:
        - Not meant to be used for training, since the shuffling of training data
            results in large structural deviations between subsequent training samples.
        - Not transferable between different molecule conformations or varying atom
            indexing.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        neighbor_list: Transform,
        nbh_transforms: Optional[List[torch.nn.Module]] = None,
        cutoff_skin: float = 0.3,
    ):
        """
        Args:
            neighbor_list: the neighbor list to use
            nbh_transforms: transforms for manipulating the neighbor lists
                provided by neighbor_list
            cutoff_skin: float
                If no atom has moved more than cutoff_skin/2 since the neighbor list
                has been updated the last time, then the neighbor list is reused.
                This will save some expensive rebuilds of the list.
        """

        super().__init__()

        self.neighbor_list = neighbor_list
        self.cutoff = neighbor_list._cutoff
        self.cutoff_skin = cutoff_skin
        self.neighbor_list._cutoff = self.cutoff + cutoff_skin
        self.nbh_transforms = nbh_transforms or []
        self.distance_calculator = spk.atomistic.PairwiseDistances()
        self.previous_inputs = {}

    # @timeit
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        update_required, inputs = self._update(inputs)
        inputs = self.distance_calculator(inputs)
        inputs = self._remove_neighbors_in_skin(inputs)

        return inputs

    def reset(self):
        self.previous_inputs = {}

    def _remove_neighbors_in_skin(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        Rij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        offsets = inputs[properties.offsets]

        rij = torch.norm(inputs[properties.Rij], dim=-1)
        cidx = torch.nonzero(rij <= self.cutoff).squeeze(-1)

        inputs[properties.Rij] = Rij[cidx]
        inputs[properties.idx_i] = idx_i[cidx]
        inputs[properties.idx_j] = idx_j[cidx]
        inputs[properties.offsets] = offsets[cidx]

        return inputs

    def _update(self, inputs):
        """Make sure the list is up-to-date."""

        # get sample index
        sample_idx = inputs[properties.idx].item()

        # check if previous neighbor list exists and make sure that this is not the
        # first update step
        if sample_idx in self.previous_inputs.keys():
            # load previous inputs
            previous_inputs = self.previous_inputs[sample_idx]
            # extract previous structure
            previous_positions = np.array(previous_inputs[properties.R], copy=True)
            previous_cell = np.array(
                previous_inputs[properties.cell].view(3, 3), copy=True
            )
            previous_pbc = np.array(previous_inputs[properties.pbc], copy=True)
            # extract current structure
            positions = inputs[properties.R]
            cell = inputs[properties.cell].view(3, 3)
            pbc = inputs[properties.pbc]
            # check if structure change is sufficiently small to reuse previous neighbor
            # list
            if (
                (previous_pbc == pbc.numpy()).any()
                and (previous_cell == cell.numpy()).any()
                and ((previous_positions - positions.numpy()) ** 2).sum(1).max()
                < 0.25 * self.cutoff_skin**2
            ):
                # reuse previous neighbor list
                inputs[properties.idx_i] = previous_inputs[properties.idx_i].clone()
                inputs[properties.idx_j] = previous_inputs[properties.idx_j].clone()
                inputs[properties.offsets] = previous_inputs[properties.offsets].clone()
                return False, inputs

        # build new neighbor list
        inputs = self._build(inputs)
        return True, inputs

    def _build(self, inputs):

        # apply all transforms to obtain new neighbor list
        inputs = self.neighbor_list(inputs)
        for nbh_transform in self.nbh_transforms:
            inputs = nbh_transform(inputs)

        # store new reference conformation and remove old one
        sample_idx = inputs[properties.idx].item()
        stored_inputs = {
            properties.R: inputs[properties.R].detach().clone(),
            properties.cell: inputs[properties.cell].detach().clone(),
            properties.pbc: inputs[properties.pbc].detach().clone(),
            properties.idx_i: inputs[properties.idx_i].detach().clone(),
            properties.idx_j: inputs[properties.idx_j].detach().clone(),
            properties.offsets: inputs[properties.offsets].detach().clone(),
        }
        self.previous_inputs.update({sample_idx: stored_inputs})

        return inputs


class TorchNeighborList(NeighborListTransform):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni

    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    References:
        https://github.com/aiqm/torchani/blob/master/torchani/aev.py
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


class FilterNeighbors(Transform):
    """
    Filter out all neighbor list indices corresponding to interactions between a set of
    atoms. This set of atoms must be specified in the input data.
    """

    def __init__(self, selection_name: str):
        """
        Args:
            selection_name (str): key in the input data corresponding to the set of
                atoms between which no interactions should be considered.
        """
        self.selection_name = selection_name
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        n_neighbors = inputs[properties.idx_i].shape[0]
        slab_indices = inputs[self.selection_name].tolist()
        kept_nbh_indices = []
        for nbh_idx in range(n_neighbors):
            i = inputs[properties.idx_i][nbh_idx].item()
            j = inputs[properties.idx_j][nbh_idx].item()
            if i not in slab_indices or j not in slab_indices:
                kept_nbh_indices.append(nbh_idx)

        inputs[properties.idx_i] = inputs[properties.idx_i][kept_nbh_indices]
        inputs[properties.idx_j] = inputs[properties.idx_j][kept_nbh_indices]
        inputs[properties.offsets] = inputs[properties.offsets][kept_nbh_indices]

        return inputs


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
        Using the neighbors contained within the cutoff shell, generate all unique pairs
        of neighbors and convert them to index arrays. Applied to the neighbor arrays,
        these arrays generate the indices involved in the atom triples.

        Example:
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
            sorted: Set to false if chosen neighbor list yields unsorted center indices
                (idx_i).
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

from typing import Dict

import torch

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CollectAtomTriples, NeighborListTransform

__all__ = ["NeighborListMD"]


class NeighborListMD:
    """
    Wrapper for neighbor list transforms to make them suitable for molecular dynamics simulations. Introduces handling
    of multiple replicas and a cutoff shell (buffer region) to avoid recomputations of the neighbor list in every step.

    The work is done by :class:`~schnetpack.transform.BatchNeighborList`, shared with batchwise structure relaxation
    framework.
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_shell: float,
        base_nbl: NeighborListTransform,
        requires_triples: bool = False,
        collate_fn: callable = _atoms_collate_fn,
    ):
        """

        Args:
            cutoff (float): Cutoff radius.
            cutoff_shell (float): Buffer region. Atoms can move this much unitil neighbor list needs to be recomputed.
            base_nbl (schnetpack.transform.NeighborListTransform): basic SchNetPack neighbor list transform.
            requires_triples (bool): Compute atom triples, e.g. for angles (default=False).
            collate_fn (callable): Collate function for batch generation. Used to combine neighbor lists of differnt
                                   replicas and molecules.
        """
        self.cutoff = cutoff
        self.cutoff_shell = cutoff_shell
        self.cutoff_full = cutoff + cutoff_shell
        self.requires_triples = requires_triples
        self._collate = collate_fn

        self.neighbor_list = BatchNeighborList(
            neighbor_list=base_nbl(cutoff),
            cutoff_skin=cutoff_shell,
            transforms=[CollectAtomTriples()] if requires_triples else None,
        )

    def get_neighbors(self, inputs: dict[str, torch.Tensor]):
        """
        Compute neighbor indices from positions and simulations cells.

        Args:
            inputs (dict(str, torch.Tensor)): input batch.

        Returns:
            dict(str, torch.Tensor): indices of neighbors, and nothing else -- the caller
            merges them into the batch it already holds.
        """
<<<<<<< HEAD
        # TODO: check consistent wrapping
        atom_types = inputs[properties.Z]
        positions = inputs[properties.R]
        n_atoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        cells = inputs[properties.cell]
        pbc = inputs[properties.pbc]

        n_molecules = n_atoms.shape[0]

        # Check which molecular environments need to be updated
        update_required = self._update_required(positions, cells, idx_m, n_molecules)

        if torch.any(update_required):
            # if updated, store current positions and cells for future comparisons
            self.previous_positions = positions.clone()
            self.previous_cells = cells.clone()

        # Split everything into individual structures
        input_batch = self._split_batch(
            atom_types, positions, n_atoms, cells, pbc, n_molecules
        )

        # Set batch construct
        if self.molecular_indices is None:
            self.molecular_indices = [{} for _ in range(n_molecules)]

        # Check which molecule needs to be updated and compute neighborhoods
        for idx in range(n_molecules):
            if update_required[idx]:
                # Get neighbors and if necessary triple indices
                self.molecular_indices[idx] = self.transform(input_batch[idx])

                # Remove superfluous entries before aggregation
                del self.molecular_indices[idx][properties.R]
                del self.molecular_indices[idx][properties.Z]
                del self.molecular_indices[idx][properties.cell]
                del self.molecular_indices[idx][properties.pbc]

        neighbor_idx = self._collate(self.molecular_indices)
        # Remove n_atoms
        del neighbor_idx[properties.n_atoms]

        # Move everything to correct device
        neighbor_idx = {p: neighbor_idx[p].to(positions.device) for p in neighbor_idx}

        # filter out all pairs in the buffer zone
        neighbor_idx = self._filter_indices(positions, neighbor_idx)

        return neighbor_idx

    def _filter_indices(
        self, positions: torch.Tensor, neighbor_idx: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Routine for filtering out pair indices and offets due to the buffer region, which would otherwise slow down
        the calculators.

        Args:
            positions (torch.Tensor): Tensor of the Cartesian atom positions.
            neighbor_idx (dict(str, torch.Tensor)): Dictionary containing pair indices and offets

        Returns:
            dict(str, torch.Tensor): Dictionary containing updated pair indices and offets
        """
        offsets = neighbor_idx[properties.offsets]
        idx_i = neighbor_idx[properties.idx_i]
        idx_j = neighbor_idx[properties.idx_j]

        Rij = positions[idx_j] - positions[idx_i] + offsets
        d_ij = torch.linalg.norm(Rij, dim=1)
        d_ij_filter = d_ij <= self.cutoff

        neighbor_idx[properties.idx_i] = neighbor_idx[properties.idx_i][d_ij_filter]
        neighbor_idx[properties.idx_j] = neighbor_idx[properties.idx_j][d_ij_filter]
        neighbor_idx[properties.offsets] = neighbor_idx[properties.offsets][
            d_ij_filter, :
        ]

        return neighbor_idx

    @staticmethod
    def _split_batch(
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        n_atoms: torch.Tensor,
        cells: torch.Tensor,
        pbc: torch.Tensor,
        n_molecules: int,
    ) -> list[dict[str, torch.tensor]]:
        """
        Split the tensors containing molecular information into the different molecules for neighbor list computation.
        Args:
            atom_types (torch.Tensor): Atom type tensor.
            positions (torch.Tensor): Atomic positions.
            n_atoms (torch.Tensor): Number of atoms in each molecule.
            cells (torch.Tensor): Simulation cells.
            pbc (torch.Tensor): Periodic boundary conditions used for each molecule.
            n_molecules (int): Number of molecules.

        Returns:
            list(dict(str, torch.Tensor))): List of input dictionaries for each molecule.
        """
        input_batch = []

        idx_c = 0
        for idx_mol in range(n_molecules):
            curr_n_atoms = n_atoms[idx_mol]
            inputs = {
                properties.n_atoms: torch.tensor([curr_n_atoms]).cpu(),
                properties.Z: atom_types[idx_c : idx_c + curr_n_atoms].cpu(),
                properties.R: positions[idx_c : idx_c + curr_n_atoms].cpu(),
            }

            if cells is None:
                inputs[properties.cell] = None
                inputs[properties.pbc] = None
            else:
                inputs[properties.cell] = cells[idx_mol].cpu()
                inputs[properties.pbc] = pbc[idx_mol].cpu()

            idx_c += curr_n_atoms
            input_batch.append(inputs)

        return input_batch
=======
        return self.neighbor_list.neighbors(inputs)
>>>>>>> 09b3a3d3 (md and bw optimizer share nbh list, atomsconverter back to master, fixed skin bug in md nbh list, fixed triplets bug)

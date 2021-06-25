import torch
import torch.nn as nn

from schnetpack.transform import (
    NeighborListTransform,
    CollectAtomTriples,
    ASENeighborList,
    TorchNeighborList,
)
from schnetpack.data.loader import _atoms_collate_fn
from typing import List, Dict
from schnetpack import properties

__all__ = ["NeighborListMD", "ASENeighborListMD", "TorchNeighborListMD"]


class NeighborListMD:
    def __init__(
        self,
        cutoff: float,
        cutoff_shell: float,
        requires_triples: bool,
        base_nbl: NeighborListTransform,
        collate_fn: callable = _atoms_collate_fn,
    ):
        self.cutoff = cutoff
        self.cutoff_shell = cutoff_shell
        self.cutoff_full = cutoff + cutoff_shell
        self.requires_triples = requires_triples
        self._collate = collate_fn

        # Build neighbor list transform
        self.transform = [base_nbl(self.cutoff_full, return_offset=True)]

        if self.requires_triples:
            self.transform.append(CollectAtomTriples())

        self.transform = nn.Sequential(*self.transform)

        # Previous cells and positions for determining update
        self.previous_positions = None
        self.previous_cells = None
        self.molecular_indices = None

    def _update_required(self, positions, cells, idx_m: torch.tensor, n_molecules: int):

        if self.previous_positions is None:
            # Everything needs to be updated
            update_required = torch.ones(n_molecules, device=idx_m.device).bool()
        elif n_molecules != len(self.molecular_indices):
            self.molecular_indices = None
            update_required = torch.ones(n_molecules, device=idx_m.device).bool()
        else:
            # Check for changes is positions
            update_positions = (
                torch.norm(self.previous_positions - positions, dim=1)
                > self.cutoff_shell
            ).float()

            # Map to individual molecules
            update_required = torch.zeros(n_molecules, device=idx_m.device).float()
            update_required = update_required.index_add(
                0, idx_m, update_positions
            ).bool()

            # Check for cell changes (is no cells are required, this will always be zero)
            update_cells = torch.any((self.previous_cells != cells).view(-1, 9), dim=1)
            update_required = torch.logical_or(update_required, update_cells)

        return update_required

    def get_neighbors(self, inputs: Dict[str, torch.Tensor]):
        # TODO: check if this is better or building Rij after the full indices have been generated
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
            else:
                self._update_Rij(input_batch[idx], idx)

        neighbor_idx = self._collate(self.molecular_indices)
        # Remove offsets and n_atoms
        del neighbor_idx[properties.offsets]
        del neighbor_idx[properties.n_atoms]

        # Move everything to correct device
        neighbor_idx = {p: neighbor_idx[p].to(positions.device) for p in neighbor_idx}

        # Make sure Rij has the right dtype (e.g. ASE nbl likes to change precision for some strange reason)
        neighbor_idx[properties.Rij] = neighbor_idx[properties.Rij].to(positions.dtype)

        return neighbor_idx

    def _update_Rij(self, inputs: Dict[str, torch.tensor], mol_idx):
        R = inputs[properties.R]
        idx_i = self.molecular_indices[mol_idx][properties.idx_i]
        idx_j = self.molecular_indices[mol_idx][properties.idx_j]

        new_Rij = R[idx_j] - R[idx_i]

        cell = inputs[properties.cell]

        if cell is not None:
            offsets = self.molecular_indices[mol_idx][properties.offsets].to(cell.dtype)
            new_Rij = new_Rij + offsets.mm(cell)

        self.molecular_indices[mol_idx][properties.Rij] = new_Rij

    @staticmethod
    def _split_batch(
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        n_atoms: torch.Tensor,
        cells: torch.Tensor,
        pbc: torch.Tensor,
        n_molecules: int,
    ) -> List[Dict[str, torch.tensor]]:
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


class ASENeighborListMD(NeighborListMD):
    def __init__(
        self, cutoff: float, cutoff_shell: float, requires_triples: bool = False
    ):
        super(ASENeighborListMD, self).__init__(
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
            requires_triples=requires_triples,
            base_nbl=ASENeighborList,
            collate_fn=_atoms_collate_fn,
        )


class TorchNeighborListMD(NeighborListMD):
    def __init__(
        self, cutoff: float, cutoff_shell: float, requires_triples: bool = False
    ):
        super(TorchNeighborListMD, self).__init__(
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
            requires_triples=requires_triples,
            base_nbl=TorchNeighborList,
            collate_fn=_atoms_collate_fn,
        )

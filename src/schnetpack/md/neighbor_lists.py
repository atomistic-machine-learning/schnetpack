"""
Neighbor lists are used to obtain the indices of neighbors surrounding an atom
for the :obj:`schnetpack.md.calculators.SchnetPackCalculator`.
Currently only a primitive version of a neighbor list is implemented, which cannot deal
with periodic boundary conditions and does not possess optimal scaling for large systems.
"""

import torch


class MDNeighborList:
    """
    Basic neighbor list template for molecular dynamics simulations required for the calculator. This is used to obtain
    the indices of the atoms adjacent to a central atom and e.g. used to compute the molecular interactions.
    The neighbor mask is zero for interactions which should not be counted and one otherwise.

    Args:
        system (object): System class containing all molecules and their replicas.
        cutoff (float): Cutoff radius used for neighbor list construction.
    """

    def __init__(self, system, cutoff):
        self.system = system
        self.cutoff = cutoff

        self.neighbor_list = None
        self.neighbor_mask = None
        self._construct_neighbor_list()

    def _construct_neighbor_list(self):
        """
        Instructions to construct the neighbor list. Needs to be defined and has to populate the neighbor_list
        and neighbor_mask tensors.

        Both, neighbor_list and neighbor_mask, should be a torch.Tensor with the dimensions:
            n_replicas x n_molecules x n_atoms x n_neighbors
        """
        raise NotImplementedError

    def update_neighbors(self):
        """
        Recompute the neighbor list (e.g. during MD simulations).
        """
        raise NotImplementedError

    def get_neighbors(self):
        """
        Convenience routine to obtain the neighbor list and neighbor mask in one step.

        Returns:
            tuple: Contains the neighbor list and neighbor mask tensors.
        """
        return self.neighbor_list, self.neighbor_mask


class SimpleNeighborList(MDNeighborList):
    """
    Basic implementation of a neighbor list. Simply enumerates the neighbors of all atoms in the molecule after
    eliminating self interactions. This work fine for small to medium sized systems, but should not be used for
    extended molecules, etc. The cutoff fulfills no purpose in this basic implementation.

    Args:
        system (object): System class containing all molecules and their replicas.
        cutoff (float): Cutoff radius used for neighbor list construction, not used in the present implementation.
    """

    def __init__(self, system, cutoff=None):
        super(SimpleNeighborList, self).__init__(system, cutoff)

    def _construct_neighbor_list(self):
        """
        Sets up a basic neighbor list and neighbor mask. Both are torch.Tensor objects of the dimension:
            n_replicas x n_molecules x n_atoms x n_neighbors
        and simply enumerate all neighbors (neighbor_list) or mask nonsensical entries due to either different cutoff
        radii or zero-padding arising from molecules of different size (neighbor_mask).
        """
        # Construct basic, unmasked tile
        basic_tile = torch.arange(self.system.max_n_atoms, device=self.system.device)[
            None, :
        ].repeat(self.system.max_n_atoms, 1)
        # Remove self interactions
        diagonal_mask = torch.eye(self.system.max_n_atoms, device=self.system.device)
        basic_tile = basic_tile[diagonal_mask != 1].view(
            self.system.max_n_atoms, self.system.max_n_atoms - 1
        )

        # Tile neighbor lists and mask to get replica and molecule dimensions
        neighbors_list = basic_tile[None, None, :, :].repeat(
            self.system.n_replicas, self.system.n_molecules, 1, 1
        )

        # Construct the neighbor mask as an outer product of the atom masks, where self interactions are removed
        self.neighbor_mask = (
            self.system.atom_masks.transpose(2, 3)[..., 1:] * self.system.atom_masks
        )

        # Multiply neighbor list by mask to remove superfluous entries
        self.neighbor_list = neighbors_list * self.neighbor_mask.long()

    def update_neighbors(self):
        """
        Simply rebuilds the neighbor list in this naive implementation.
        """
        self._construct_neighbor_list()

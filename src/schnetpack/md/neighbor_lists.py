"""
Neighbor lists are used to obtain the indices of neighbors surrounding an atom
for the :obj:`schnetpack.md.calculators.SchnetPackCalculator`.
Currently only a primitive version of a neighbor list is implemented, which cannot deal
with periodic boundary conditions and does not possess optimal scaling for large systems.
"""

import torch
import schnetpack


class MDNeighborList:
    """
    Basic neighbor list template for molecular dynamics simulations required for the calculator. This is used to obtain
    the indices of the atoms adjacent to a central atom and e.g. used to compute the molecular interactions.
    The neighbor mask is zero for interactions which should not be counted and one otherwise.

    Args:
        cutoff (float): Cutoff radius used for neighbor list construction.
    """

    def __init__(self, system, cutoff, shell=None):

        self.system = system

        self.cutoff = cutoff
        self.shell = shell

        if shell is not None:
            self.cutoff_shell = cutoff + shell
        else:
            self.cutoff_shell = cutoff

        self.neighbor_list = None
        self.neighbor_mask = None
        self.offsets = None
        self.max_neighbors = None

    def _construct_neighbor_list(self):
        """
        Instructions to construct the neighbor list. Needs to be defined and has to populate the neighbor_list
        and neighbor_mask tensors.

        Both, neighbor_list and neighbor_mask, should be a torch.Tensor with the dimensions:
            n_replicas x n_molecules x n_atoms x n_neighbors
        """
        raise NotImplementedError

    def _update_required(self):
        """
        Function to determine whether the neighbor list should be recomputed for the system. This could e.g. be
        based on the maximum distance all atoms moved, etc. The return value should be True if it needs to be
        recomputed and False otherwise.

        Returns:
            bool: Indicator whether it is necessary to compute a new neighbor list or not.
        """
        raise NotImplementedError

    def get_neighbors(self):
        """
        Convenience routine to obtain the neighbor list and neighbor mask in one step.

        Returns:
            tuple: Contains the neighbor list and neighbor mask tensors.
        """
        if self._update_required() or self.neighbor_list is None:
            self._construct_neighbor_list()

        return self.neighbor_list, self.neighbor_mask, self.offsets


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
        # Set the maximum neighbors to include all interactions
        self.max_neighbors = self.system.max_n_atoms - 1

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

    def _update_required(self):
        """
        Since all interatomic distances are computed by default, the neighbor list never has to be updated.

        Returns:
            bool: Indicator whether it is necessary to compute a new neighbor list or not.
        """
        return False


class EnvironmentProviderNeighborList(MDNeighborList):
    """

    Args:
        system (schnetpack.md.System):
        cutoff (float):
        shell (float):
    """

    def __init__(self, system, cutoff=5.0, shell=1.0):
        super(EnvironmentProviderNeighborList, self).__init__(system, cutoff + shell)
        self.shell = shell

        # Store last positions and cells for determining whether the neighbor list needs to be recomputed
        self.last_positions = None
        self.last_cells = None

        # Setup the environment provider
        self._environment_provider = None
        self._set_environment_provider()

    def _set_environment_provider(self):
        """

        Returns:

        """
        raise NotImplementedError

    def _construct_neighbor_list(self):
        """

        Returns:

        """
        atoms = self.system.get_ase_atoms()

        neighbor_idx = []
        offsets = []
        max_neighbors = 0

        for mol in atoms:
            nbh_idx, offset = self._environment_provider.get_environment(mol)
            neighbor_idx.append(nbh_idx)
            offsets.append(offset)
            print(offset.shape, "SH")
            max_neighbors = max(max_neighbors, nbh_idx.shape[1])

        print(max_neighbors)

        self.neighbor_list = -torch.ones(
            self.system.n_replicas,
            self.system.n_molecules,
            self.system.max_n_atoms,
            max_neighbors,
            device=self.system.device,
        ).long()
        self.offsets = torch.zeros(
            self.system.n_replicas,
            self.system.n_molecules,
            self.system.max_n_atoms,
            max_neighbors,
            3,
            device=self.system.device,
        )

        count = 0
        for r_idx in range(self.system.n_replicas):
            for m_idx in range(self.system.n_molecules):
                n_atoms = self.system.n_atoms[m_idx]
                n_nbh = neighbor_idx[count].shape[1]
                # TODO: DEBUG HERE!!! WHAT IS WITH NEIGHBOR DIMENSION IN OFFSETS?
                self.neighbor_list[r_idx, m_idx, :n_atoms, :n_nbh] = torch.from_numpy(
                    neighbor_idx[count]
                )
                self.offsets[r_idx, m_idx, :n_atoms, :n_nbh, :] = torch.from_numpy(
                    offsets[count]
                )
                count += 1
                # print(count)

        print("ME HERE")

        self.max_neighbors = max_neighbors
        self.neighbor_mask = torch.zeros_like(self.neighbor_list)
        self.neighbor_mask[self.neighbor_list >= 0] = 1.0

        # Mask away -1 indices for invalid atoms, since they do not work with torch.gather
        self.neighbor_list = self.neighbor_list * self.neighbor_mask.long()

        # Since the list was recomputed, update old cells and positions
        self.last_positions = self.system.positions.clone().detach()
        self.last_cells = self.system.cells.clone().detach()

        print(self.offsets.shape, "NL OFF")
        print(self.neighbor_list.shape, "NL NL")
        print(self.neighbor_mask.shape, "NL NBM")
        # exit()

    def _update_required(self):
        """

        Returns:

        """
        # TODO: In theory this is not needed.
        # Compute the neighbor list if these two are not set
        if self.last_positions is None or self.last_cells is None:
            return True

        # Check if cell has changed
        if not torch.allclose(self.last_cells, self.system.cells):
            return True

        # Check if atoms have moved out of the boundary
        max_displacement = torch.max(
            torch.norm(self.system.positions - self.last_positions, 2, 3)
        ).detach()
        if max_displacement >= self.shell:
            return True

        return False


class ASENeighborList(EnvironmentProviderNeighborList):
    def __init__(self, system, cutoff=5.0, shell=1.0):
        super(ASENeighborList, self).__init__(system, cutoff=cutoff, shell=shell)

    def _set_environment_provider(self):
        self._environment_provider = schnetpack.environment.AseEnvironmentProvider(
            self.cutoff
        )


class TorchNeighborList(EnvironmentProviderNeighborList):
    def __init__(self, system, cutoff=5.0, shell=1.0):
        super(TorchNeighborList, self).__init__(system, cutoff=cutoff, shell=shell)

    def _set_environment_provider(self):
        self._environment_provider = schnetpack.environment.TorchEnvironmentProvider(
            self.cutoff
        )

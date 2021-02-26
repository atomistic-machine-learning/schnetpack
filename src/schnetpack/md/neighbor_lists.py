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
        shell (float): Buffer region around the cutoff radius. A new neighbor list is only constructed if an atom
                       moves farther than this shell. (Or if the simulation cell changes.)
    """

    def __init__(self, cutoff, shell=None, device=None):
        self.device = device

        # Check cutoff and shell, as well as possible conventions
        self.cutoff = cutoff
        self.shell = shell

        if self.cutoff is not None:
            if self.shell is not None:
                self.cutoff_shell = self.cutoff + self.shell
            else:
                self.cutoff_shell = self.cutoff
        else:
            self.cutoff_shell = None

        # Init basic containers
        self.neighbor_list = None
        self.neighbor_mask = None
        self.offsets = None
        self.max_neighbors = None

        # Store last positions and cells for determining whether the neighbor list needs to be recomputed
        self.last_positions = None
        self.last_cells = None

    def get_neighbors(self, system):
        """
        Convenience routine to obtain the neighbor list and neighbor mask in one step.

        Returns:
            tuple: Contains the neighbor list and neighbor mask tensors.
        """
        if self._update_required(system) or self.neighbor_list is None:
            self._construct_neighbor_list(system)

        return self.neighbor_list, self.neighbor_mask, self.offsets

    def _construct_neighbor_list(self, system):
        """
        Instructions to construct the neighbor list. Needs to be defined and has to populate the neighbor_list
        and neighbor_mask tensors.

        Both, neighbor_list and neighbor_mask, should be a torch.Tensor with the dimensions:
            n_replicas x n_molecules x n_atoms x n_neighbors
        """
        raise NotImplementedError

    def _update_required(self, system):
        """
        Function to determine whether the neighbor list should be recomputed for the system. This could e.g. be
        based on the maximum distance all atoms moved, etc. The return value should be True if it needs to be
        recomputed and False otherwise.

        Returns:
            bool: Indicator whether it is necessary to compute a new neighbor list or not.
        """
        # Compute the neighbor list if these two are not set
        if self.last_positions is None or self.last_cells is None:
            return True

        # Check if cell has changed
        if not torch.allclose(self.last_cells, system.cells):
            return True

        # Check if atoms have moved out of the boundary
        max_displacement = torch.max(
            torch.norm(system.positions - self.last_positions, 2, 3)
        ).detach()
        if max_displacement >= self.shell:
            return True

        return False


class SimpleNeighborList(MDNeighborList):
    """
    Basic implementation of a neighbor list. Simply enumerates the neighbors of all atoms in the molecule after
    eliminating self interactions. This work fine for small to medium sized systems, but should not be used for
    extended molecules, etc. The cutoff fulfills no purpose in this basic implementation. This neighbor list should
    never be used in combination with periodic boundary conditions.

    Args:
        system (object): System class containing all molecules and their replicas.
        cutoff (float): Cutoff radius used for neighbor list construction, not used in the present implementation.
    """

    def __init__(self, cutoff=None, shell=None, device=None):
        super(SimpleNeighborList, self).__init__(cutoff, shell, device=device)

    def _construct_neighbor_list(self, system):
        """
        Sets up a basic neighbor list, neighbor mask and offset array. The first two are torch.Tensor objects of the
        dimension: n_replicas x n_molecules x n_atoms x n_neighbors. The offsets have the dimension
        n_replicas x n_molecules x n_atoms x n_neighbors x 3.
        This neighbor list simply enumerates all neighbors (neighbor_list) or mask nonsensical entries due to either
        different cutoff radii or zero-padding arising from molecules of different size (neighbor_mask).
        """
        # Set the maximum neighbors to include all interactions
        self.max_neighbors = system.max_n_atoms - 1

        # Construct basic, unmasked tile
        basic_tile = torch.arange(system.max_n_atoms, device=system.device)[
            None, :
        ].repeat(system.max_n_atoms, 1)
        # Remove self interactions
        diagonal_mask = torch.eye(system.max_n_atoms, device=system.device)
        basic_tile = basic_tile[diagonal_mask != 1].view(
            system.max_n_atoms, system.max_n_atoms - 1
        )

        # Tile neighbor lists and mask to get replica and molecule dimensions
        neighbors_list = basic_tile[None, None, :, :].repeat(
            system.n_replicas, system.n_molecules, 1, 1
        )

        # Construct the neighbor mask as an outer product of the atom masks, where self interactions are removed
        self.neighbor_mask = (
            system.atom_masks.transpose(2, 3)[..., 1:] * system.atom_masks
        )

        # Construct cell offsets
        if system.cells is not None:
            self.offsets = torch.zeros(
                system.n_replicas,
                system.n_molecules,
                system.max_n_atoms,
                self.max_neighbors,
                3,
                device=system.device,
            )
        else:
            self.offsets = None

        # Multiply neighbor list by mask to remove superfluous entries
        self.neighbor_list = neighbors_list * self.neighbor_mask.long()

    def _update_required(self, system):
        """
        Since all interatomic distances are computed by default, the neighbor list never has to be updated.

        Returns:
            bool: Indicator whether it is necessary to compute a new neighbor list or not.
        """
        return False


class EnvironmentProviderNeighborList(MDNeighborList):
    """
    Basic neighbor list class to be used with the environment providers with SchNetPack. The corresponding provider
    needs to be set in the `_set_environment_provider` function. Since this currently operates sequentially, it will
    provide suboptimal performance for systems with many replicas and/or molecules.

    Args:
        cutoff (float): Cutoff radius used for neighbor list construction.
        shell (float): Buffer region around the cutoff radius. A new neighbor list is only constructed if an atom
                       moves farther than this shell. (Or if the simulation cell changes.)
    """

    def __init__(self, cutoff, shell=1.0, device=None, use_internal_units=True):
        super(EnvironmentProviderNeighborList, self).__init__(
            cutoff, shell, device=device
        )
        self.use_internal_units = use_internal_units
        self.provider_cutoff = self._get_provider_cutoff()

        # Setup the environment provider
        self._environment_provider = None
        self._set_environment_provider()

    def _get_provider_cutoff(self):
        if self.use_internal_units:
            provider_cutoff = self.cutoff_shell
        else:
            provider_cutoff = self.cutoff_shell / schnetpack.md.MDUnits.angs2internal
        return provider_cutoff

    def _set_environment_provider(self):
        """
        This function is intended to set the environment provider in neighbor lists based on this class.
        """
        raise NotImplementedError

    def _construct_neighbor_list(self, system):
        """
        Construct the neighbor list using an environment provider. Since all providers are based on ASE atoms objects,
        these objects are first extracted from the system. Then the neighbor lists ae constructed sequentially and
        reconverted into the format required for the calculators. In addition, the old cells and positons are
        stored to check if updates of the neighbor list are necessary.
        """
        atoms = system.get_ase_atoms(internal_units=self.use_internal_units)

        neighbor_idx = []
        offsets = []
        max_neighbors = 0

        for mol in atoms:
            nbh_idx, offset = self._environment_provider.get_environment(mol)
            neighbor_idx.append(nbh_idx)
            offsets.append(offset)
            max_neighbors = max(max_neighbors, nbh_idx.shape[1])

        self.neighbor_list = -torch.ones(
            system.n_replicas,
            system.n_molecules,
            system.max_n_atoms,
            max_neighbors,
            device=system.device,
        ).long()

        self.offsets = torch.zeros(
            system.n_replicas,
            system.n_molecules,
            system.max_n_atoms,
            max_neighbors,
            3,
            device=system.device,
        )

        count = 0
        for r_idx in range(system.n_replicas):
            for m_idx in range(system.n_molecules):
                n_atoms = system.n_atoms[m_idx]
                n_nbh = neighbor_idx[count].shape[1]
                self.neighbor_list[r_idx, m_idx, :n_atoms, :n_nbh] = torch.from_numpy(
                    neighbor_idx[count]
                )

                if system.cells is not None:
                    self.offsets[r_idx, m_idx, :n_atoms, :n_nbh, :] = torch.from_numpy(
                        offsets[count]
                    )
                else:
                    self.offsets = None

                count += 1

        self.max_neighbors = max_neighbors
        self.neighbor_mask = torch.zeros_like(self.neighbor_list)
        self.neighbor_mask[self.neighbor_list >= 0] = 1.0

        # Mask away -1 indices for invalid atoms, since they do not work with torch.gather
        self.neighbor_list = self.neighbor_list * self.neighbor_mask.long()

        # Since the list was recomputed, update old cells and positions
        self.last_positions = system.positions.clone().detach()

        if system.cells is not None:
            self.last_cells = system.cells.clone().detach()


class ASENeighborList(EnvironmentProviderNeighborList):
    """
    Neighbor list based on the schnetpack.utils.environment.AseEnvironmentProvider. This can deal with periodic
    boundary conditions and general unit cells. However, the provider runs on CPU only and will only provide
    significant performance gains over the torch based one for very large systems.
    The ASE neighbor_list internally uses a minimum bin size of 3A, hence positions and cells need to be converted
    to A before passing them to the neighbor list to avoid performance issues.

    Args:
        cutoff (float): Cutoff radius used for neighbor list construction.
        shell (float): Buffer region around the cutoff radius. A new neighbor list is only constructed if an atom
                       moves farther than this shell. (Or if the simulation cell changes.)
    """

    def __init__(self, cutoff, shell, device=None):
        super(ASENeighborList, self).__init__(
            cutoff=cutoff, shell=shell, device=device, use_internal_units=False
        )

    def _set_environment_provider(self):
        """
        Set the environment provider.
        """
        self._environment_provider = schnetpack.environment.AseEnvironmentProvider(
            self.provider_cutoff
        )


class TorchNeighborList(EnvironmentProviderNeighborList):
    """
    Neighbor list based on the schnetpack.utils.environment.TorchEnvironmentProvider. For moderately sized systems
    with cells/periodic boundary conditions this should have a good performance.

    Args:
        cutoff (float): Cutoff radius used for neighbor list construction.
        shell (float): Buffer region around the cutoff radius. A new neighbor list is only constructed if an atom
                       moves farther than this shell. (Or if the simulation cell changes.)
        device (torch.device): Device used when computing the neighbor list.
    """

    def __init__(self, cutoff, shell, device=torch.device("cpu")):
        super(TorchNeighborList, self).__init__(
            cutoff=cutoff, shell=shell, device=device
        )

    def _set_environment_provider(self):
        """
        Set the environment provider.
        """
        self._environment_provider = schnetpack.environment.TorchEnvironmentProvider(
            self.cutoff_shell, self.device
        )


class DualNeighborList:
    def __init__(
        self, cutoff_short, cutoff_long, neighbor_list, shell=1.0, device=None
    ):
        self.neighbor_list_short = neighbor_list(cutoff_short, shell, device=device)
        self.neighbor_list_long = neighbor_list(cutoff_long, shell, device=device)

    def get_neighbors(self, system):
        neighbors, neighbor_mask, offsets = self.neighbor_list_short.get_neighbors(
            system
        )
        return neighbors, neighbor_mask, offsets

    def get_neighbors_lr(self, system):
        (
            neighbors_long,
            neighbor_mask_long,
            offsets_long,
        ) = self.neighbor_list_long.get_neighbors(system)
        return neighbors_long, neighbor_mask_long, offsets_long

    @property
    def max_neighbors(self):
        return self.neighbor_list_short.max_neighbors

    @property
    def max_neighbors_lr(self):
        return self.neighbor_list_long.max_neighbors

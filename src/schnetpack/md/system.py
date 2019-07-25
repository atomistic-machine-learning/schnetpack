"""
This module is used to store all information on the simulated atomistic systems.
It includes functionality for loading molecules from files.
All this functionality is encoded in the :obj:`schnetpack.md.System` class.
"""
import torch
from ase.io import read

from schnetpack.md.utils import MDUnits, compute_centroid, batch_inverse
from schnetpack.md.neighbor_lists import SimpleNeighborList


class SystemException(Exception):
    pass


class System:
    """
    Container for all properties associated with the simulated molecular system
    (masses, positions, momenta, ...). Uses atomic units internally.

    In order to simulate multiple systems efficiently dynamics properties
    (positions, momenta, forces) are torch tensors with the following
    dimensions:
        n_replicas x n_molecules x n_atoms x 3

    Here n_replicas is the number of copies for every molecule. In a normal
    simulation, these are treated as independent molecules e.g. for sampling
    purposes. In the case of ring polymer molecular dynamics (using the
    RingPolymer integrator), these replicas correspond to the beads of the
    polymer. n_molecules is the number of different molecules constituting
    the system, these can e.g. be different initial configurations of the
    same system (once again for sampling) or completely different molecules. In
    the latter case, the maximum number of atoms n_atoms (3rd dimension) is
    determined automatically and all arrays padded with zeros.

    Static properties (n_atoms, masses, atom_types and atom_masks) are stored in
    tensors of the shape:
        n_atoms : 1 x n_molecules (the same for all replicas)
        masses : 1 x n_molecules x n_atoms x 1 (the same for all replicas)
        atom_types : n_replicas x n_molecules x n_atoms x 1 (are brought to this
                     shape in order to avoid reshapes during every calculator
                     call)
        atom_masks : n_replicas x n_molecules x n_atoms x 1 (can change if
                     neighbor lists change for the replicas)

    n_atoms contains the number of atoms present in every molecule, masses
    and atom_types contain the molcular masses and nuclear charges.
    atom_masks is a binary array used to mask superfluous entries introduced
    by the zero-padding for differently sized molecules.

    Finally a dictionary properties stores the results of every calculator
    call for easy access of e.g. energies and dipole moments.

    Args:
        n_replicas (int): Number of replicas generated for each molecule.
        device (str): Computation device (default='cuda').
        neighborlist (object): Routine for generating the neighbor list used
                               in the calculator (default is
                               SimpleNeighborList).
    """

    # TODO: Introduce periodic boundary conditions

    def __init__(
        self,
        n_replicas,
        device="cuda",
        neighborlist=SimpleNeighborList,
        initializer=None,
    ):

        # Specify device
        self.device = device

        # number of molecules, replicas of each and vector with the number of
        # atoms in each molecule
        self.n_replicas = n_replicas
        self.n_molecules = None
        self.n_atoms = None
        self.max_n_atoms = None

        # General static molecular properties
        self.atom_types = None
        self.masses = None
        self.atom_masks = None

        # Dynamic properties updated during simulation
        self.positions = None
        self.momenta = None
        self.forces = None

        # Property dictionary, updated during simulation
        self.properties = {}

        # Initialize neighbor list for the calculator
        self.neighbor_list = neighborlist

        # Initialize initial conditions
        self.initializer = initializer

    def load_molecules_from_xyz(self, path_to_file):
        """
        Wrapper for loading molecules from .xyz file

        Args:
            path_to_file (str): path to data-file

        """
        molecules = read(path_to_file)
        if not type(molecules) == list:
            molecules = [molecules]
        self.load_molecules(molecules=molecules)

    def load_molecules(self, molecules):
        """
        Initializes all required variables and tensors based on a list of ASE
        atoms objects.

        Args:
            molecules list(ase.Atoms): List of ASE atoms objects containing
            molecular structures and chemical elements.
        """

        # 1) Get maximum number of molecules, number of replicas and number of
        #    overall systems
        self.n_molecules = len(molecules)

        # 2) Construct array with number of atoms in each molecule
        self.n_atoms = torch.zeros(
            self.n_molecules, dtype=torch.long, device=self.device
        )

        for i in range(self.n_molecules):
            self.n_atoms[i] = molecules[i].get_number_of_atoms()

        # 3) Determine the maximum number of atoms present (in case of
        #    differently sized molecules)
        self.max_n_atoms = int(torch.max(self.n_atoms))

        # 4) Construct basic properties and masks
        self.atom_types = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, device=self.device
        ).long()
        self.atom_masks = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, device=self.device
        )
        self.masses = torch.ones(self.n_molecules, self.max_n_atoms, device=self.device)

        # Relevant for dynamic properties: positions, momenta, forces
        self.positions = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, 3, device=self.device
        )
        self.momenta = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, 3, device=self.device
        )

        # 5) Populate arrays according to the data provided in molecules
        for i in range(self.n_molecules):
            # Static properties
            self.atom_types[:, i, : self.n_atoms[i]] = torch.from_numpy(
                molecules[i].get_atomic_numbers()
            )
            self.atom_masks[:, i, : self.n_atoms[i]] = 1.0
            self.masses[i, : self.n_atoms[i]] = torch.from_numpy(
                molecules[i].get_masses() * MDUnits.d2amu
            )

            # Dynamic properties
            self.positions[:, i, : self.n_atoms[i], :] = torch.from_numpy(
                molecules[i].positions * MDUnits.angs2bohr
            )

        # 6) Do proper broadcasting here for easier use in e.g. integrators and
        #    thermostats afterwards
        self.masses = self.masses[None, :, :, None]
        self.atom_masks = self.atom_masks[..., None]

        # 7) Build neighbor lists
        if self.neighbor_list is not None:
            self.neighbor_list = self.neighbor_list(self)

        # 8) Initialize Momenta
        if self.initializer:
            self.initializer.initialize_system(self)

    @property
    def center_of_mass(self):
        """
        Compute the center of mass for each replica and molecule

        Returns:
            torch.Tensor: n_replicas x n_molecules x 1 x 3 tensor holding the
                          center of mass.
        """
        # Mask mass array
        masses = self.masses * self.atom_masks
        # Compute center of mass
        center_of_mass = torch.sum(self.positions * masses, 2) / torch.sum(masses, 2)
        return center_of_mass

    def remove_com(self):
        """
        Move all structures to their respective center of mass.
        """
        # Mask to avoid offsets
        self.positions -= self.center_of_mass[:, :, None, :]
        # Apply atom masks to avoid artificial shifts
        self.positions *= self.atom_masks

    def remove_com_translation(self):
        """
        Remove all components in the current momenta associated with
        translational motion.
        """
        self.momenta -= (
            torch.sum(self.momenta, 2, keepdim=True)
            / self.n_atoms.float()[None, :, None, None]
        )
        # Apply atom masks to avoid artificial shifts
        self.momenta *= self.atom_masks

    def remove_com_rotation(self, detach=True):
        """
        Remove all components in the current momenta associated with rotational
        motion using Eckart conditons.

        Args:
            detach (bool): Whether computational graph should be detached in
                           order to accelerated the simulation (default=True).
        """
        # Compute the moment of inertia tensor
        moment_of_inertia = (
            torch.sum(self.positions ** 2, 3, keepdim=True)[..., None]
            * torch.eye(3, device=self.device)[None, None, None, :, :]
            - self.positions[..., :, None] * self.positions[..., None, :]
        )
        moment_of_inertia = torch.sum(moment_of_inertia * self.masses[..., None], 2)

        # Compute the angular momentum
        angular_momentum = torch.sum(torch.cross(self.positions, self.momenta, -1), 2)

        # Compute the angular velocities
        angular_velocities = torch.matmul(
            angular_momentum[:, :, None, :], batch_inverse(moment_of_inertia)
        )

        # Compute individual atomic contributions
        rotational_velocities = torch.cross(
            angular_velocities.repeat(1, 1, self.max_n_atoms, 1), self.positions, -1
        )

        if detach:
            rotational_velocities = rotational_velocities.detach()

        # Subtract rotation from overall motion (apply atom mask)
        self.momenta -= rotational_velocities * self.masses * self.atom_masks

    @property
    def velocities(self):
        """
        Convenience property to access molecular velocities instead of the
        momenta (e.g for power spectra)

        Returns:
            torch.Tensor: Velocity tensor with the same shape as the momenta.
        """
        return self.momenta / self.masses

    @property
    def centroid_positions(self):
        """
        Convenience property to access the positions of the centroid during
        ring polymer molecular dynamics. Does not make sense during a
        standard dynamics setup.

        Returns:
            torch.Tensor: Tensor of the shape 1 x n_molecules x n_atoms x 3
            holding the centroid positions.
        """
        return compute_centroid(self.positions)

    @property
    def centroid_momenta(self):
        """
        Convenience property to access the centroid momenta during ring
        polymer molecular dynamics. Does not make sense during a standard
        dynamics setup.

        Returns:
            torch.Tensor: Tensor of the shape 1 x n_molecules x n_atoms x 3
                          holding the centroid momenta.
        """
        return compute_centroid(self.momenta)

    @property
    def centroid_velocities(self):
        """
        Convenience property to access the velocities of the centroid during
        ring polymer molecular dynamics (e.g. for computing power spectra).
        Does not make sense during a standard dynamics setup.

        Returns:
            torch.Tensor: Tensor of the shape 1 x n_molecules x n_atoms x 3
            holding the centroid velocities.
        """
        return self.centroid_momenta / self.masses

    @property
    def kinetic_energy(self):
        """
        Convenience property for computing the kinetic energy associated with
        each replica and molecule.

        Returns:
            torch.Tensor: Tensor of the kinetic energies (in Hartree) with
                          the shape n_replicas x n_molecules
        """
        # Apply atom mask
        momenta = self.momenta * self.atom_masks
        kinetic_energy = 0.5 * torch.sum(
            torch.sum(momenta ** 2, 3) / self.masses[..., 0], 2
        )
        return kinetic_energy.detach()

    @property
    def temperature(self):
        """
        Convenience property for accessing the instantaneous temperatures of
        each replica and molecule.

        Returns:
            torch.Tensor: Tensor of the instantaneous temperatures (in
                          Kelvin) with the shape n_replicas x n_molecules
        """
        temperature = (
            2.0
            / (3.0 * MDUnits.kB * self.n_atoms.float()[None, :])
            * self.kinetic_energy
        )
        return temperature

    @property
    def centroid_kinetic_energy(self):
        """
        Convenience property for computing the kinetic energy associated with
        the centroid of each molecule. Only sensible in the context of ring
        polymer molecular dynamics.

        Returns:
            torch.Tensor: Tensor of the centroid kinetic energies (in
                          Hartree) with the shape 1 x n_molecules
        """
        # Apply atom mask
        centroid_momenta = self.centroid_momenta * self.atom_masks
        kinetic_energy = 0.5 * torch.sum(
            torch.sum(centroid_momenta ** 2, 3) / self.masses[..., 0], 2
        )
        return kinetic_energy

    @property
    def centroid_temperature(self):
        """
        Convenience property for accessing the instantaneous temperatures of
        the centroid of each molecule. Only makes sense in the context of
        ring polymer molecular dynamics.

        Returns:
            torch.Tensor: Tensor of the instantaneous centroid temperatures (
                          in Kelvin) with the shape 1 x n_molecules
        """
        temperature = (
            2.0
            / (3.0 * self.n_atoms.float()[None, :] * MDUnits.kB)
            * self.centroid_kinetic_energy
        )
        return temperature

    @property
    def state_dict(self):
        """
        State dict for storing the system state.

        Returns:
            dict: Dictionary containing all properties for restoring the
                  current state of the system during simulation.
        """
        state_dict = {
            "positions": self.positions,
            "momenta": self.momenta,
            "forces": self.forces,
            "properties": self.properties,
            "n_atoms": self.n_atoms,
            "atom_types": self.atom_types,
            "masses": self.masses,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        """
        Routine for restoring the state of a system specified in a previously
        stored state dict. Used to restart molecular dynamics simulations.

        Args:
            state_dict (dict): State dict of the system state.
        """
        self.positions = state_dict["positions"]
        self.momenta = state_dict["momenta"]
        self.forces = state_dict["forces"]
        self.properties = state_dict["properties"]
        self.n_atoms = state_dict["n_atoms"]
        self.atom_types = state_dict["atom_types"]
        self.masses = state_dict["masses"]

        self.n_replicas = self.positions.shape[0]
        self.n_molecules = self.positions.shape[1]
        self.max_n_atoms = self.positions.shape[2]

        # Build atom masks according to the present number of atoms
        for i in range(self.n_molecules):
            self.atom_masks[:, i, : self.n_atoms[i], :] = 1.0

        # Rebuild neighbor lists with new system specifications
        if self.neighbor_list is not None:
            self.neighbor_list.__init__(self)

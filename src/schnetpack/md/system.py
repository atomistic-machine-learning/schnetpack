"""
This module is used to store all information on the simulated atomistic systems.
It includes functionality for loading molecules from files.
All this functionality is encoded in the :obj:`schnetpack.md.System` class.
"""
import torch

from schnetpack.utils import int2precision
from ase import Atoms

from typing import Union, List

from schnetpack import units as spk_units

__all__ = ["System"]


class SystemException(Exception):
    pass


class System:
    """
    Container for all properties associated with the simulated molecular system
    (masses, positions, momenta, ...). Uses MD unit system defined in `schnetpack.units` internally.

    In order to simulate multiple systems efficiently dynamics properties
    (positions, momenta, forces) are torch tensors with the following
    dimensions:
        n_replicas x (n_molecules * n_atoms) x 3

    Here n_replicas is the number of copies for every molecule. In a normal
    simulation, these are treated as independent molecules e.g. for sampling
    purposes. In the case of ring polymer molecular dynamics (using the
    RingPolymer integrator), these replicas correspond to the beads of the
    polymer. n_molecules is the number of different molecules constituting
    the system, these can e.g. be different initial configurations of the
    same system (once again for sampling) or completely different molecules.
    Atoms of multiple molecules are concatenated.

    Static properties are stored in tensors of the shape:
        n_atoms : n_molecules (the same for all replicas)
        masses : 1 x (n_molecules * n_atoms) x 1 (the same for all replicas)
        atom_types : (n_molecules * n_atoms)
        index_m : (n_molecules * n_atoms)

    `n_atoms` contains the number of atoms present in every molecule, `masses`
    and `atom_types` contain the molcular masses and nuclear charges.
    `index_m` is an index for mapping atoms to individual molecules.

    Finally a dictionary properties stores the results of every calculator
    call for easy access of e.g. energies and dipole moments.

    Args:
        device (str, torch.device): Computation device (default='cuda').
        precision (int, torch.dtype): Precision used for floating point numbers (default=32).
    """

    # Index for aggregation
    total_n_atoms = None
    index_m = None

    # number of molecules, replicas of each and vector with the number of
    # atoms in each molecule
    n_replicas = None
    n_molecules = None
    n_atoms = None

    # General static molecular properties
    atom_types = None
    masses = None

    # Dynamic properties updated during simulation
    positions = None
    momenta = None
    forces = None
    energies = None

    # Properties for periodic boundary conditions and crystal cells
    cells = None
    pbc = None
    stress = None  # Used for the computation of the pressure

    # Property dictionary, updated during simulation
    properties = {}

    def __init__(self, device="cuda", precision=32):
        # Specify device
        self._device = device

        # specify numerical precision
        self._precision = int2precision(precision)

    def load_molecules(
        self,
        molecules: Union[Atoms, List[Atoms]],
        n_replicas: int = 1,
        position_unit_input: Union[str, float] = "Angstrom",
        mass_unit_input: Union[str, float] = 1.0,
    ):
        """
        Initializes all required variables and tensors based on a list of ASE
        atoms objects.

        Args:
            molecules (ase.Atoms, list(ase.Atoms)): List of ASE atoms objects containing
                molecular structures and chemical elements.
            n_replicas (int): Number of replicas (e.g. for RPMD)
            position_unit_input (str, float): Position units of the input structures (default="Angstrom")
            mass_unit_input (str, float): Units of masses passed in the ASE atoms. Assumed to be Dalton.
        """
        self.n_replicas = n_replicas

        # TODO: make cells/PBC False if not set?

        # Set up unit conversion
        positions2internal = spk_units.convert_units(
            position_unit_input, spk_units.length
        )
        mass2internal = spk_units.convert_units(mass_unit_input, spk_units.mass)

        # 0) Check if molecules is a single ase.Atoms object and wrap it in list.
        if isinstance(molecules, Atoms):
            molecules = [molecules]

        # 1) Get number of molecules, number of replicas and number of
        #    overall systems
        self.n_molecules = len(molecules)

        # 2) Construct array with number of atoms in each molecule
        self.n_atoms = torch.zeros(self.n_molecules, dtype=torch.long)

        for i in range(self.n_molecules):
            self.n_atoms[i] = molecules[i].get_global_number_of_atoms()

        # 3) Get total n_molecule x n_atom dimension
        self.total_n_atoms = torch.sum(self.n_atoms).item()
        # initialize index vector for aggregation
        self.index_m = torch.zeros(self.total_n_atoms, dtype=torch.long)

        # 3) Construct basic property arrays
        self.atom_types = torch.zeros(self.total_n_atoms, dtype=torch.long)
        self.masses = torch.ones(1, self.total_n_atoms, 1)

        # Relevant for dynamic properties: positions, momenta, forces
        self.positions = torch.zeros(self.n_replicas, self.total_n_atoms, 3)
        self.momenta = torch.zeros(self.n_replicas, self.total_n_atoms, 3)
        self.forces = torch.zeros(self.n_replicas, self.total_n_atoms, 3)

        self.energies = torch.zeros(self.n_replicas, self.n_molecules, 1)

        # Relevant for periodic boundary conditions and simulation cells
        self.cells = torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        self.stress = torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        self.pbc = torch.zeros(1, self.n_molecules, 3)

        # 5) Populate arrays according to the data provided in molecules
        idx_c = 0
        for i in range(self.n_molecules):
            n_atoms = self.n_atoms[i]

            # Aggregation array
            self.index_m[idx_c : idx_c + n_atoms] = i

            # Static properties
            self.atom_types[idx_c : idx_c + n_atoms] = torch.from_numpy(
                molecules[i].get_atomic_numbers()
            )
            self.masses[0, idx_c : idx_c + n_atoms, 0] = torch.from_numpy(
                molecules[i].get_masses() * mass2internal
            )

            # Dynamic properties
            self.positions[:, idx_c : idx_c + n_atoms, :] = torch.from_numpy(
                molecules[i].positions * positions2internal
            )

            # Properties for cell simulations
            self.cells[:, i, :, :] = torch.from_numpy(
                molecules[i].cell.array * positions2internal
            )
            self.pbc[0, i, :] = torch.from_numpy(molecules[i].pbc)

            idx_c += n_atoms

        # Convert periodic boundary conditions to Boolean tensor
        self.pbc = self.pbc.bool()

        # Check for cell/pbc stuff:
        if torch.sum(torch.abs(self.cells)) == 0.0:
            if torch.sum(self.pbc) > 0.0:
                raise SystemException("Found periodic boundary conditions but no cell.")
            else:
                self.cells = None

        # Move everything to device and precision
        self.device = self.device
        self.precision = self.precision

    def _sum_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for summing atomic contributions for each molecule.

        Args:
            x (torch.Tensor): Input tensor of the shape ( : x (n_molecules * n_atoms) x ...)

        Returns:
            torch.Tensor: Aggregated tensor of the shape ( : x n_molecules x ...)
        """
        x_tmp = torch.zeros(
            self.n_replicas,
            self.n_molecules,
            *x.shape[2:],
            device=self._device,
            dtype=self._precision
        )
        return x_tmp.index_add(1, self.index_m, x)

    def _mean_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for computing mean over atomic contributions for each molecule.

        Args:
            x (torch.Tensor): Input tensor of the shape ( : x (n_molecules * n_atoms) x ...)

        Returns:
            torch.Tensor: Aggregated tensor of the shape ( : x n_molecules x ...)
        """
        return self._sum_atoms(x) / self.n_atoms[None, :, None]

    def _expand_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for expanding molecular contributions over the corresponding atoms.

        Args:
            x (torch.Tensor): Tensor of the shape ( : x n_molecules x ...)

        Returns:
            torch.Tensor: Tensor of the shape ( : x (n_molecules * n_atoms) x ...)
        """
        return x[:, self.index_m, ...]

    @property
    def center_of_mass(self):
        """
        Compute the center of mass for each replica and molecule

        Returns:
            torch.Tensor: n_replicas x n_molecules x 3 tensor holding the
                          center of mass.
        """
        # Compute center of mass
        center_of_mass = self._sum_atoms(
            self.positions * self.masses
        ) / self._sum_atoms(self.masses)
        return center_of_mass

    def remove_center_of_mass(self):
        """
        Move all structures to their respective center of mass.
        """
        self.positions -= self._expand_atoms(self.center_of_mass)

    def remove_translation(self):
        """
        Remove all components in the current momenta associated with
        translational motion.
        """
        self.momenta -= self._expand_atoms(self._mean_atoms(self.momenta))

    def remove_com_rotation(self):
        """
        Remove all components in the current momenta associated with rotational
        motion using Eckart conditions.
        """
        # Compute the moment of inertia tensor
        moment_of_inertia = (
            torch.sum(self.positions ** 2, dim=2, keepdim=True)[..., None]
            * torch.eye(3, device=self._device, dtype=self._precision)[None, None, :, :]
            - self.positions[..., :, None] * self.positions[..., None, :]
        )

        moment_of_inertia = self._sum_atoms(moment_of_inertia * self.masses[..., None])

        # Compute the angular momentum
        angular_momentum = self._sum_atoms(
            torch.cross(self.positions, self.momenta, -1)
        )

        # Compute the angular velocities
        angular_velocities = torch.matmul(
            angular_momentum[:, :, None, :], torch.inverse(moment_of_inertia)
        ).squeeze(2)

        # Compute individual atomic contributions
        rotational_velocities = torch.cross(
            self._expand_atoms(angular_velocities), self.positions, -1
        )

        # Subtract rotation from overall motion (apply atom mask)
        self.momenta -= rotational_velocities * self.masses

    def get_ase_atoms(self, position_unit_output="Angstrom"):
        # TODO: make sensible unit conversion and update docs
        """
        Convert the stored molecular configurations into ASE Atoms objects. This is e.g. used for the
        neighbor lists based on environment providers. All units are atomic units by default, as used in the calculator

        Args:
            position_unit_output (str, float): Target units for position output.

        Returns:
            list(ase.Atoms): List of ASE Atoms objects, with the replica and molecule dimension flattened.
        """
        internal2positions = spk_units.convert_units(
            spk_units.length, position_unit_output
        )

        atoms = []
        for idx_r in range(self.n_replicas):
            idx_c = 0
            for idx_m in range(self.n_molecules):
                n_atoms = self.n_atoms[idx_m]

                positions = (
                    self.positions[idx_r, idx_c : idx_c + n_atoms]
                    .cpu()
                    .detach()
                    .numpy()
                ) * internal2positions

                atom_types = (
                    self.atom_types[idx_c : idx_c + n_atoms].cpu().detach().numpy()
                )

                if self.cells is not None:
                    cell = (
                        self.cells[idx_r, idx_m].cpu().detach().numpy()
                        * internal2positions
                    )
                    pbc = self.pbc[0, idx_m].cpu().detach().numpy()
                else:
                    cell = None
                    pbc = None

                mol = Atoms(atom_types, positions, cell=cell, pbc=pbc)
                atoms.append(mol)

                idx_c += n_atoms

        return atoms

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
    def kinetic_energy(self):
        """
        Convenience property for computing the kinetic energy associated with
        each replica and molecule.

        Returns:
            torch.Tensor: Tensor of the kinetic energies (in Hartree) with
                          the shape n_replicas x n_molecules x 1
        """
        kinetic_energy = 0.5 * self._sum_atoms(
            torch.sum(self.momenta ** 2, dim=2, keepdim=True) / self.masses
        )
        return kinetic_energy

    @property
    def kinetic_energy_tensor(self):
        """
        Compute the kinetic energy tensor (outer product of momenta divided by masses) for pressure computation.
        The standard kinetic energy is the trace of this tensor.

        Returns:
            torch.tensor: n_replicas x n_molecules x 3 x 3 tensor containing kinetic energy components.

        """
        # Apply atom mask
        kinetic_energy_tensor = 0.5 * self._sum_atoms(
            self.momenta[..., None]
            * self.momenta[:, :, None, :]
            / self.masses[..., None]
        )
        return kinetic_energy_tensor.detach()

    @property
    def temperature(self):
        """
        Convenience property for accessing the instantaneous temperatures of
        each replica and molecule.

        Returns:
            torch.Tensor: Tensor of the instantaneous temperatures (in
                          Kelvin) with the shape n_replicas x n_molecules x 1
        """
        temperature = (
            2.0
            / (3.0 * spk_units.kB * self.n_atoms[None, :, None])
            * self.kinetic_energy
        )
        return temperature

    @property
    def centroid_positions(self):
        """
        Convenience property to access the positions of the centroid during
        ring polymer molecular dynamics. Does not make sense during a
        standard dynamics setup.

        Returns:
            torch.Tensor: Tensor of the shape 1 x (n_molecules * n_atoms) x 3
            holding the centroid positions.
        """
        return torch.mean(self.positions, dim=0, keepdim=True)

    @property
    def centroid_momenta(self):
        """
        Convenience property to access the centroid momenta during ring
        polymer molecular dynamics. Does not make sense during a standard
        dynamics setup.

        Returns:
            torch.Tensor: Tensor of the shape 1 x (n_molecules * n_atoms) x 3
                          holding the centroid momenta.
        """
        return torch.mean(self.momenta, dim=0, keepdim=True)

    @property
    def centroid_velocities(self):
        """
        Convenience property to access the velocities of the centroid during
        ring polymer molecular dynamics (e.g. for computing power spectra).
        Does not make sense during a standard dynamics setup.

        Returns:
            torch.Tensor: Tensor of the shape (1 x n_molecules * n_atoms) x 3
            holding the centroid velocities.
        """
        return self.centroid_momenta / self.masses

    @property
    def centroid_kinetic_energy(self):
        """
        Convenience property for computing the kinetic energy associated with
        the centroid of each molecule. Only sensible in the context of ring
        polymer molecular dynamics.

        Returns:
            torch.Tensor: Tensor of the centroid kinetic energies (in
                          Hartree) with the shape 1 x n_molecules x 1
        """
        kinetic_energy = 0.5 * self._sum_atoms(
            torch.sum(self.centroid_momenta ** 2, dim=2, keepdim=True) / self.masses
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
                          in Kelvin) with the shape 1 x n_molecules x 1
        """
        temperature = (
            2.0
            / (3.0 * spk_units.kB * self.n_atoms[None, :, None])
            * self.centroid_kinetic_energy
        )
        return temperature

    @property
    def volume(self):
        """
        Compute the cell volumes if cells are present.

        Returns:
            torch.tensor: n_replicas x n_molecules x 1 containing the volumes.
        """
        if self.cells is None:
            return None
        else:
            volume = torch.sum(
                self.cells[:, :, 0]
                * torch.cross(self.cells[:, :, 1], self.cells[:, :, 2], dim=2),
                dim=2,
                keepdim=True,
            )
            return volume

    def compute_pressure(self, tensor=False, kinetic_component=False):
        """
        Compute the pressure (tensor) based on the stress tensor of the systems.

        Args:
            tensor (bool): Instead of a scalar pressure, return the full pressure tensor. (Required for
                           anisotropic cell deformation.)
            kinetic_component (bool): Include the kinetic energy component during the computation of the
                                      pressure (default=False).

        Returns:
            torch.Tensor: Depending on the tensor-flag, returns a tensor containing the pressure with dimensions
                          n_replicas x n_molecules x 1 (False) or n_replicas x n_molecules x 3 x 3 (True).
        """
        if self.cells is None:
            raise SystemError(
                "Simulation cell and stress required for computation of the instantaneous pressure."
            )

        # TODO: check how often kinetic component is actually required
        # TODO: is kinetic energy tensor actually needed elsewhere?

        pressure = -self.stress

        if tensor:
            if kinetic_component:
                pressure += 2 * self.kinetic_energy_tensor / self.volume[..., None]
        else:
            pressure = torch.einsum("abii->ab", pressure)[..., None] / 3.0
            if kinetic_component:
                pressure += 2.0 * self.kinetic_energy / self.volume / 3.0

        return pressure

    def wrap_positions(self, eps=1e-6):
        """
        Move atoms outside the box back into the box for all dimensions with periodic boundary
        conditions.

        Args:
            eps (float): Small offset for numerical stability
        """
        pbc_atomic = self._expand_atoms(self.pbc)

        # Compute fractional coordinates
        inverse_cell = torch.inverse(self.cells)
        inverse_cell = self._expand_atoms(inverse_cell)
        inv_positions = torch.sum(self.positions[..., None] * inverse_cell, dim=2)

        # Get periodic coordinates
        periodic = torch.masked_select(inv_positions, pbc_atomic)

        # Apply periodic boundary conditions (with small buffer)
        periodic = periodic + eps
        periodic = periodic % 1.0
        periodic = periodic - eps

        # Update fractional coordinates
        inv_positions.masked_scatter_(pbc_atomic, periodic)

        # Convert to positions
        self.positions = torch.sum(
            inv_positions[..., None] * self._expand_atoms(self.cells), dim=2
        )

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        """
        Setter function for precision. This automatically converts integers to their `torch.dtype` counterparts and
        converts all float tensors to the target precision.

        Args:
            precision (int, torch.dtype: Either integer (e.g. 32) or `torch.dtype` (e.g. `torch.float32`)
        """
        self._precision = int2precision(precision)

        self.atom_types = self.atom_types.to(self._precision)
        self.masses = self.masses.to(self._precision)
        self.positions = self.positions.to(self._precision)
        self.momenta = self.momenta.to(self._precision)
        self.forces = self.forces.to(self._precision)
        self.energies = self.energies.to(self._precision)
        self.stress = self.stress.to(self._precision)

        if self.cells is not None:
            self.cells = self.cells.to(self._precision)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        """
        Setter function for device. This automatically moves all relevant tensors to the target device.

        Args:
            device (str, torch.device): Target device
        """
        self._device = device

        self.index_m = self.index_m.to(self._device)
        self.n_atoms = self.n_atoms.to(self._device)
        self.atom_types = self.atom_types.to(self._device)
        self.masses = self.masses.to(self._device)
        self.positions = self.positions.to(self._device)
        self.momenta = self.momenta.to(self._device)
        self.forces = self.forces.to(self._device)
        self.energies = self.energies.to(self._device)
        self.pbc = self.pbc.to(self._device)
        self.stress = self.stress.to(self._device)

        if self.cells is not None:
            self.cells = self.cells.to(self._device)

    @property
    def state_dict(self):
        """
        State dict for storing the system state.

        Returns:
            dict: Dictionary containing all properties for restoring the
                  current state of the system during simulation.
        """
        state_dict = {
            "index_m": self.index_m,
            "positions": self.positions,
            "momenta": self.momenta,
            "forces": self.forces,
            "properties": self.properties,
            "n_atoms": self.n_atoms,
            "atom_types": self.atom_types,
            "masses": self.masses,
            "cells": self.cells,
            "pbc": self.pbc,
            "stress": self.stress,
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
        self.index_m = state_dict["index_m"]
        self.positions = state_dict["positions"]
        self.momenta = state_dict["momenta"]
        self.forces = state_dict["forces"]
        self.energies = state_dict["energies"]
        self.properties = state_dict["properties"]
        self.n_atoms = state_dict["n_atoms"]
        self.atom_types = state_dict["atom_types"]
        self.masses = state_dict["masses"]
        self.cells = state_dict["cells"]
        self.pbc = state_dict["pbc"]
        self.stress = state_dict["stress"]

        self.n_replicas = self.positions.shape[0]
        self.total_n_atoms = self.positions.shape[1]
        self.n_molecules = self.n_atoms.shape[0]

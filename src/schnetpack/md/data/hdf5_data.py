"""
Class for extracting information from the HDF5 files generated during simulation by the
:obj:`schnetpack.md.simulation_hooks.logging_hooks.FileLogger`.
In addition to loading structures, velocities, etc., various postprocessing functions are available.
"""

import json
import logging
import h5py
import numpy as np
from ase import Atoms
from typing import Optional
from tqdm import trange

from schnetpack import properties, units

log = logging.getLogger(__name__)


class HDF5LoaderError(Exception):
    """
    Exception for HDF5 loader class.
    """

    pass


class HDF5Loader:
    """
    Class for loading HDF5 datasets written by the FileLogger. By default, this requires at least a MoleculeStream to be
    present. PropertyData is also read by default, but can be disabled.

    Args:
        hdf5_database (str): Path to the database file.
        skip_initial (int): Skip the initial N configurations in the trajectory, e.g. to account for equilibration
                            (default=0).
        load_properties (bool): Extract and reconstruct the property data stored by a PropertyStream (e.g. forces,
                                energies, etc.), enabled by default.
    """

    def __init__(
        self,
        hdf5_database: str,
        skip_initial: Optional[int] = 0,
        load_properties: Optional[bool] = True,
    ):
        self.database = h5py.File(hdf5_database, "r", swmr=True, libver="latest")
        self.skip_initial = skip_initial
        self.data_groups = list(self.database.keys())

        self.properties = {}

        # Load basic structure properties and MD info
        if "molecules" not in self.data_groups:
            raise HDF5LoaderError(
                "Molecule data not found in {:s}".format(hdf5_database)
            )
        else:
            self._load_molecule_data()

        # If requested, load other properties predicted by the model stored via PropertyStream
        if load_properties:
            if "properties" not in self.data_groups:
                raise HDF5LoaderError(
                    "Molecule properties not found in {:s}".format(hdf5_database)
                )
            else:
                self._load_property_data()

        # Do formatting for info
        loaded_properties = list(self.properties.keys())
        if len(loaded_properties) == 1:
            loaded_properties = str(loaded_properties[0])
        else:
            loaded_properties = (
                ", ".join(loaded_properties[:-1]) + " and " + loaded_properties[-1]
            )

        log.info(
            "Loaded properties {:s} from {:s}".format(loaded_properties, hdf5_database)
        )

    def _load_molecule_data(self):
        """
        Load data stored by a MoleculeStream. This contains basic information on the system and is required to be
        present for the Loader.
        """
        # This is for molecule streams
        structures = self.database["molecules"]

        # General database info
        self.n_replicas = structures.attrs["n_replicas"]
        self.n_molecules = structures.attrs["n_molecules"]
        self.total_n_atoms = structures.attrs["total_n_atoms"]
        self.n_atoms = structures.attrs["n_atoms"]
        self.time_step = structures.attrs["time_step"]

        # Set up molecule ranges
        self.molecule_range = np.pad(np.cumsum(self.n_atoms), (1, 0), mode="constant")

        # Determine loading
        self.total_entries = structures.attrs["entries"]
        self.entries = self.total_entries - self.skip_initial

        # Get atom types and masses
        self.properties[properties.Z] = structures.attrs["atom_types"]
        self.properties[properties.masses] = structures.attrs["masses"]

        # Get position and velocity blocks
        raw_positions = structures[self.skip_initial : self.total_entries]

        entry_width = self.total_n_atoms * 3

        # Extract energies
        entry_start = 0
        entry_stop = self.n_molecules
        self.properties[f"{properties.energy}_system"] = raw_positions[
            :, :, entry_start:entry_stop
        ].reshape(self.entries, self.n_replicas, self.n_molecules)

        # Extract positions
        entry_start = entry_stop
        entry_stop += entry_width
        self.properties[properties.R] = raw_positions[
            :, :, entry_start:entry_stop
        ].reshape(self.entries, self.n_replicas, self.total_n_atoms, 3)

        # Extract velocities if present
        if structures.attrs["has_velocities"]:
            entry_start = entry_stop
            entry_stop += entry_width
            self.properties["velocities"] = raw_positions[
                :, :, entry_start:entry_stop
            ].reshape(self.entries, self.n_replicas, self.total_n_atoms, 3)

        # Extract quantities due to simulation in box
        if structures.attrs["has_cells"]:
            # Get simulation cells
            entry_start = entry_stop
            entry_stop += 9 * self.n_molecules
            self.properties[properties.cell] = raw_positions[
                :, :, entry_start:entry_stop
            ].reshape(self.entries, self.n_replicas, self.n_molecules, 3, 3)

            # Get stress tensor
            entry_start = entry_stop
            entry_stop += 9 * self.n_molecules
            self.properties[f"{properties.stress}_system"] = raw_positions[
                :, :, entry_start:entry_stop
            ].reshape(self.entries, self.n_replicas, self.n_molecules, 3, 3)
        else:
            self.properties[properties.cell] = None

        self.pbc = structures.attrs["pbc"]

    def _load_property_data(self):
        """
        Load properties and their shape from the corresponding group in the hdf5 database.
        Properties are then reshaped to original form and stores in the self.properties dictionary.
        """
        # And for property stream
        properties = self.database["properties"]
        property_shape = json.loads(properties.attrs["shapes"])
        property_positions = json.loads(properties.attrs["positions"])

        # Reformat properties
        for prop in property_positions:
            prop_pos = slice(*property_positions[prop])
            self.properties[prop] = properties[
                self.skip_initial : self.total_entries, :, prop_pos
            ].reshape(
                (
                    self.total_entries - self.skip_initial,
                    self.n_replicas,
                    *property_shape[prop],
                )
            )

    def get_property(
        self,
        property_name: str,
        atomistic: bool,
        mol_idx: Optional[int] = 0,
        replica_idx: Optional[int] = None,
    ):
        """
        Extract property from dataset.

        Args:
            property_name (str): Name of the property as contained in the self.properties dictionary.
            atomistic (bool): Whether the property is atomistic (e.g. forces) or defined for the whole molecule
                              (e.g. energies, dipole moments).
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x property dimensions array containing the requested property collected during
                      the simulation.
        """
        if atomistic:
            mol_idx = slice(
                self.molecule_range[mol_idx], self.molecule_range[mol_idx + 1]
            )
        else:
            mol_idx = mol_idx

        # Check whether property is present
        if property_name not in self.properties:
            raise HDF5LoaderError(f"Property {property_name} not found in database.")

        if self.properties[property_name] is None:
            # Typically used for cells
            return None
        elif property_name == properties.Z or property_name == properties.masses:
            # Special case for atom types and masses
            return self.properties[property_name][mol_idx]
        else:
            # Standard properties
            target_property = self.properties[property_name][:, :, mol_idx, ...]

        # Compute the centroid unless requested otherwise
        if replica_idx is None:
            if target_property is not None:
                target_property = np.mean(target_property, axis=1)
        else:
            if target_property is not None:
                target_property = target_property[:, replica_idx, ...]

        return target_property

    def get_velocities(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Auxiliary routine for getting the velocities of specific molecules and replicas.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x N_atoms x 3 array containing the atom velocities of the simulation in internal units.
        """
        return self.get_property(
            "velocities", atomistic=True, mol_idx=mol_idx, replica_idx=replica_idx
        )

    def get_positions(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Auxiliary routine for getting the positions of specific molecules and replicas.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x N_atoms x 3 array containing the atom positions of the simulation in internal units.
        """
        return self.get_property(
            properties.R, atomistic=True, mol_idx=mol_idx, replica_idx=replica_idx
        )

    def get_kinetic_energy(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Auxiliary routine for computing the kinetic energy of every configuration based on it's velocities.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps array containing the kinetic energy of every configuration in internal units.
        """
        # Get the velocities
        velocities = self.get_velocities(mol_idx=mol_idx, replica_idx=replica_idx)

        # Get the masses
        masses = self.get_property(
            properties.masses, atomistic=True, mol_idx=mol_idx, replica_idx=replica_idx
        )

        # Compute the kinetic energy as 1/2*m*v^2
        kinetic_energy = 0.5 * np.sum(
            masses[None, :, None] * velocities**2, axis=(1, 2)
        )

        return kinetic_energy

    def get_potential_energy(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Auxiliary routine for extracting a systems potential energy.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps array containing the potential energy of every configuration in internal units.
        """
        energy_key = f"{properties.energy}_system"
        return self.get_property(
            energy_key, atomistic=False, mol_idx=mol_idx, replica_idx=replica_idx
        )

    def get_temperature(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Auxiliary routine for computing the instantaneous temperature of every configuration.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps array containing the temperature of every configuration in Kelvin.
        """
        # Get the velocities
        # Get the kinetic energy
        kinetic_energy = self.get_kinetic_energy(
            mol_idx=mol_idx, replica_idx=replica_idx
        )

        # Compute the temperature
        temperature = 2.0 / (3.0 * units.kB * self.n_atoms[mol_idx]) * kinetic_energy

        return temperature

    def get_volume(self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None):
        """
        Auxiliary routine for computing the cell volume in periodic simulations.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps array containing the cell volume of every configuration in internal units.
        """
        cells = self.get_property(
            properties.cell, atomistic=False, mol_idx=mol_idx, replica_idx=replica_idx
        )
        return np.linalg.det(cells)

    def get_stress(self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None):
        """
        Auxiliary routine for extracting the stress tensor in cell simulations.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x 3 x 3 array containing the stress tensor of every configuration in internal units.
        """
        stress_key = f"{properties.stress}_system"
        return self.get_property(
            stress_key, atomistic=False, mol_idx=mol_idx, replica_idx=replica_idx
        )

    def get_pressure(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Auxiliary routine for computing the pressure in periodic simulations.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps array containing the pressure of every configuration in bar.
        """
        # Get kinetic energy and volume
        kinetic_energy = self.get_kinetic_energy(
            mol_idx=mol_idx, replica_idx=replica_idx
        )
        volume = self.get_volume(mol_idx=mol_idx, replica_idx=replica_idx)

        # Compute isotropic stress
        stress = (
            np.einsum(
                "bii->b", self.get_stress(mol_idx=mol_idx, replica_idx=replica_idx)
            )
            / 3.0
        )

        # Compute the pressure and convert to bar
        pressure = (2.0 / 3.0 * kinetic_energy / volume - stress) / units.bar

        return pressure

    def convert_to_atoms(
        self, mol_idx: Optional[int] = 0, replica_idx: Optional[int] = None
    ):
        """
        Converts molecular structures to a list of ASE Atom objects. Length units are converted from the internal unit
        system to Angstrom.

        Args:
            mol_idx (optional, int):  Index of molecule to extract (default=0).
            replica_idx (optional, int): Index of replica to extract. If set to None, centroid of property is computed
                                         instead (default=None).

        Returns:
            list(ase.Atoms): List of ASE atom objects containing molecular structures.
        """
        positions = self.get_positions(
            mol_idx=mol_idx, replica_idx=replica_idx
        ) / units.unit2internal("Angstrom")

        atomic_numbers = self.get_property(
            properties.Z, atomistic=True, mol_idx=mol_idx, replica_idx=replica_idx
        )

        cells = self.get_property(
            properties.cell, atomistic=False, mol_idx=mol_idx, replica_idx=replica_idx
        )

        if cells is None:
            cells = [None] * self.entries
        else:
            cells = cells / units.unit2internal("Angstrom")

        all_atoms = []

        log.info("Extracting structures...")
        for idx in trange(self.entries):
            atoms = Atoms(
                atomic_numbers, positions[idx], cell=cells[idx], pbc=self.pbc[mol_idx]
            )
            atoms.wrap()
            all_atoms.append(atoms)

        return all_atoms

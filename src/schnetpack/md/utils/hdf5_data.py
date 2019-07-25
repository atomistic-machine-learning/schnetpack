"""
Class for extracting information from the HDF5 files generated during simulation by the
:obj:`schnetpack.md.simulation_hooks.logging_hooks.FileLogger`.
In addition to loading structures, velocities, etc., various postprocessing functions are available.
"""

import json
import logging
import h5py
import numpy as np
from ase import data

from schnetpack import Properties
from schnetpack.md.utils import MDUnits


class HDF5LoaderError(Exception):
    """
    Exception for HDF5 loader class.
    """

    pass


class HDF5Loader:
    """
    Class for loading HDF5 datasets written by the FileLogger. By default, this requires at least a MoleculeSteam to be
    present. PropertyData is also read by default, but can be disabled.

    Args:
        hdf5_database (str): Path to the database file.
        skip_initial (int): Skip the initial N configurations in the trajectory, e.g. to account for equilibration
                            (default=0).
        load_properties (bool): Extract and reconstruct the property data stored by a PropertyStream (e.g. forces,
                                energies, etc.), enabled by default.
    """

    def __init__(self, hdf5_database, skip_initial=0, load_properties=True):
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
                self._load_properties()

        # Do formatting for info
        loaded_properties = list(self.properties.keys())
        if len(loaded_properties) == 1:
            loaded_properties = str(loaded_properties[0])
        else:
            loaded_properties = (
                ", ".join(loaded_properties[:-1]) + " and " + loaded_properties[-1]
            )

        logging.info(
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
        self.n_atoms = structures.attrs["n_atoms"]
        self.total_entries = structures.attrs["entries"]
        self.time_step = structures.attrs["time_step"]
        self.entries = self.total_entries - self.skip_initial

        # Write to main property dictionary
        self.properties[Properties.Z] = structures.attrs["atom_types"][0, ...]
        self.properties[Properties.R] = structures[
            self.skip_initial : self.total_entries, ..., :3
        ]
        self.properties["velocities"] = structures[
            self.skip_initial : self.total_entries, ..., 3:
        ]

    def _load_properties(self):
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
                self.skip_initial : self.total_entries, :, :, prop_pos
            ].reshape(
                (
                    self.total_entries - self.skip_initial,
                    self.n_replicas,
                    self.n_molecules,
                    *property_shape[prop],
                )
            )

    def get_property(self, property_name, mol_idx=0, replica_idx=None, atomistic=False):
        """
        Extract property from dataset.

        Args:
            property_name (str): Name of the property as contained in the self.properties dictionary.
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.
            atomistic (bool): Whether the property is atomistic (e.g. forces) or defined for the whole molecule
                              (e.g. energies, dipole moments). If set to True, the array is masked according to the
                               number of atoms for the requested molecule to counteract potential zero-padding.
                               (default=False)

        Returns:
            np.array: N_steps x property dimensions array containing the requested property collected during the simulation.
        """

        # Special case for atom types
        if property_name == Properties.Z:
            return self.properties[Properties.Z][mol_idx]

        # Check whether property is present
        if property_name not in self.properties:
            raise HDF5LoaderError(f"Property {property_name} not found in database.")

        # Mask by number of atoms if property is declared atomistic
        if atomistic:
            n_atoms = self.n_atoms[mol_idx]
            target_property = self.properties[property_name][
                :, :, mol_idx, :n_atoms, ...
            ]
        else:
            target_property = self.properties[property_name][:, :, mol_idx, ...]

        # Compute the centroid unless requested otherwise
        if replica_idx is None:
            target_property = np.mean(target_property, axis=1)
        else:
            target_property = target_property[:, replica_idx, ...]

        return target_property

    def get_velocities(self, mol_idx=0, replica_idx=None):
        """
        Auxiliary routine for getting the velocities of specific molecules and replicas.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x N_atoms x 3 array containing the atom velocities of the simulation in atomic units.
        """
        return self.get_property(
            "velocities", mol_idx=mol_idx, replica_idx=replica_idx, atomistic=True
        )

    def get_positions(self, mol_idx=0, replica_idx=None):
        """
        Auxiliary routine for getting the positions of specific molecules and replicas.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x N_atoms x 3 array containing the atom positions of the simulation in atomic units.
        """
        return self.get_property(
            Properties.R, mol_idx=mol_idx, replica_idx=replica_idx, atomistic=True
        )

    def get_kinetic_energy(self, mol_idx=0, replica_idx=None):
        """
        Auxiliary routine for computing the kinetic energy of every configuration based on it's velocities.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps array containing the kinetic eenrgy of every configuration in atomic units.
        """
        # Get the velocities
        velocities = self.get_velocities(mol_idx=mol_idx, replica_idx=replica_idx)

        # Get the masses and convert to correct units
        masses = (
            data.atomic_masses[self.properties[Properties.Z][mol_idx]] * MDUnits.d2amu
        )

        # Compute the kinetic energy as 1/2*m*v^2
        kinetic_energy = 0.5 * np.sum(
            masses[None, :, None] * velocities ** 2, axis=(1, 2)
        )
        return kinetic_energy

    def get_temperature(self, mol_idx=0, replica_idx=None):
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
        temperature = 2.0 / (3.0 * MDUnits.kB * self.n_atoms[mol_idx]) * kinetic_energy

        return temperature

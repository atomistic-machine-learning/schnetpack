"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os
import warnings

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Dataset

import schnetpack as spk
from schnetpack import Properties
from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples

from .partitioning import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "AtomsData",
    "AtomsDataError",
    "AtomsConverter",
    "get_center_of_mass",
    "get_center_of_geometry",
]


def get_center_of_mass(atoms):
    """
    Computes center of mass.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of mass
    """
    masses = atoms.get_masses()
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()


def get_center_of_geometry(atoms):
    """
    Computes center of geometry.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of geometry
    """
    return atoms.arrays["positions"].mean(0)


class AtomsDataError(Exception):
    pass


class AtomsData(Dataset):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database. Use together with schnetpack.data.AtomsLoader to feed data
    to your model.

    Args:
        dbpath (str): path to directory containing database.
        subset (list, optional): indices to subset. Set to None for entire database.
        available_properties (list, optional): complete set of physical properties
            that are contained in the database.
        load_only (list, optional): reduced set of properties to be loaded
        units (list, optional): definition of units for all available properties
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
    """

    ENCODING = "utf-8"

    def __init__(
        self,
        dbpath,
        subset=None,
        available_properties=None,
        load_only=None,
        units=None,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        centering_function=get_center_of_mass,
    ):
        if not dbpath.endswith(".db"):
            raise AtomsDataError(
                "Invalid dbpath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        self.dbpath = dbpath

        # check if database is deprecated:
        if self._is_deprecated():
            self._deprecation_update()

        self.subset = subset
        self.load_only = load_only
        self.available_properties = self.get_available_properties(available_properties)
        if load_only is None:
            self.load_only = self.available_properties
        if units is None:
            units = [1.0] * len(self.available_properties)
        self.units = dict(zip(self.available_properties, units))
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centering_function = centering_function

    def get_available_properties(self, available_properties):
        """
        Get available properties from argument or database.

        Args:
            available_properties (list or None): all properties of the dataset

        Returns:
            (list): all properties of the dataset
        """
        # use the provided list
        if not os.path.exists(self.dbpath) or len(self) == 0:
            if available_properties is None:
                raise AtomsDataError(
                    "Please define available_properties or set "
                    "db_path to an existing database!"
                )
            return available_properties
        # read database properties
        with connect(self.dbpath) as conn:
            atmsrw = conn.get(1)
            db_properties = list(atmsrw.data.keys())
        # check if properties match
        if available_properties is None or set(db_properties) == set(
            available_properties
        ):
            return db_properties

        raise AtomsDataError(
            "The available_properties {} do not match the "
            "properties in the database {}!".format(available_properties, db_properties)
        )

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        warnings.warn(
            "create_splits is deprecated, "
            + "use schnetpack.data.train_test_split instead",
            DeprecationWarning,
        )
        return train_test_split(self, num_train, num_val, split_file)

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = (
            idx if self.subset is None or len(idx) == 0 else np.array(self.subset)[idx]
        )
        return type(self)(
            dbpath=self.dbpath,
            subset=subidx,
            load_only=self.load_only,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            available_properties=self.available_properties,
        )

    def __len__(self):
        if self.subset is None:
            with connect(self.dbpath) as conn:
                return conn.count()
        return len(self.subset)

    def __getitem__(self, idx):
        at, properties = self.get_properties(idx)
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))

        return properties

    def _subset_index(self, idx):
        # get row
        if self.subset is None:
            idx = int(idx)
        else:
            idx = int(self.subset[idx])
        return idx

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at

    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Args:
            key: Name of metadata entry. Return full dict if `None`.

        Returns:
            value: Value of metadata entry or full metadata dict, if key is `None`.

        """
        with connect(self.dbpath) as conn:
            if key is None:
                return conn.metadata
            if key in conn.metadata.keys():
                return conn.metadata[key]
        return None

    def set_metadata(self, metadata=None, **kwargs):
        """
        Sets the metadata dictionary of the ASE db.

        Args:
            metadata (dict): dictionary of metadata for the ASE db
            kwargs: further key-value pairs for convenience
        """

        # merge all metadata
        if metadata is not None:
            kwargs.update(metadata)

        with connect(self.dbpath) as conn:
            conn.metadata = kwargs

    def update_metadata(self, data):
        with connect(self.dbpath) as conn:
            metadata = conn.metadata
        metadata.update(data)
        self.set_metadata(metadata)

    def _add_system(self, conn, atoms, **properties):
        data = {}

        # add available properties to database
        for pname in self.available_properties:
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data)

    def add_system(self, atoms, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms (ase.Atoms): system composition and geometry
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.dbpath) as conn:
            self._add_system(conn, atoms, **properties)

    def add_systems(self, atoms_list, property_list):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list (list of ase.Atoms): system composition and geometry
            property_list (list): Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset.

        """
        with connect(self.dbpath) as conn:
            for at, prop in zip(atoms_list, property_list):
                self._add_system(conn, at, **prop)

    def get_properties(self, idx):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index

        Returns:

        """
        idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for pname in self.load_only:
            properties[pname] = torch.FloatTensor(row.data[pname])

        # extract/calculate structure
        properties = _convert_atoms(
            at,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            output=properties,
        )

        return at, properties

    def _get_atomref(self, property):
        """
        Returns single atom reference values for specified `property`.

        Args:
            property (str): property name

        Returns:
            list: list of atomrefs
        """
        labels = self.get_metadata("atref_labels")
        if labels is None:
            return None

        col = [i for i, l in enumerate(labels) if l == property]
        assert len(col) <= 1

        if len(col) == 1:
            col = col[0]
            atomref = np.array(self.get_metadata("atomrefs"))[:, col : col + 1]
        else:
            atomref = None

        return atomref

    def get_atomref(self, properties):
        """
        Return multiple single atom reference values as a dictionary.

        Args:
            properties (list or str): Desired properties for which the atomrefs are
                calculated.

        Returns:
            dict: atomic references
        """
        if type(properties) is not list:
            properties = [properties]
        return {p: self._get_atomref(p) for p in properties}

    def _is_deprecated(self):
        """
        Check if database is deprecated.

        Returns:
            (bool): True if ase db is deprecated.
        """
        # check if db exists
        if not os.path.exists(self.dbpath):
            return False

        # get properties of first atom
        with connect(self.dbpath) as conn:
            data = conn.get(1).data

        # check byte style deprecation
        if True in [pname.startswith("_dtype_") for pname in data.keys()]:
            return True
        # fallback for properties stored directly in the row
        if True in [type(val) != np.ndarray for val in data.values()]:
            return True

        return False

    def _deprecation_update(self):
        """
        Update deprecated database to a valid ase database.
        """
        warnings.warn(
            "The database is deprecated and will be updated automatically. "
            "The old database is moved to {}.deprecated!".format(self.dbpath)
        )

        # read old database
        atoms_list, properties_list = spk.utils.read_deprecated_database(self.dbpath)
        metadata = self.get_metadata()

        # move old database
        os.rename(self.dbpath, self.dbpath + ".deprecated")

        # write updated database
        self.set_metadata(metadata=metadata)
        with connect(self.dbpath) as conn:
            for atoms, properties in tqdm(
                zip(atoms_list, properties_list),
                "Updating new database",
                total=len(atoms_list),
            ):
                conn.write(atoms, data=properties)


def _convert_atoms(
    atoms,
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    centering_function=None,
    output=None,
):
    """
        Helper function to convert ASE atoms object to SchNetPack input format.

        Args:
            atoms (ase.Atoms): Atoms object of molecule
            environment_provider (callable): Neighbor list provider.
            collect_triples (bool, optional): Set to True if angular features are needed.
            centering_function (callable or None): Function for calculating center of
                molecule (center of mass/geometry/...). Center will be subtracted from
                positions.
            output (dict): Destination for converted atoms, if not None

    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.
    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    cell = np.array(atoms.cell.array, dtype=np.float32)  # get cell array

    inputs[Properties.Z] = torch.LongTensor(atoms.numbers.astype(np.int))
    positions = atoms.positions.astype(np.float32)
    if centering_function:
        positions -= centering_function(atoms)
    inputs[Properties.R] = torch.FloatTensor(positions)
    inputs[Properties.cell] = torch.FloatTensor(cell)

    # get atom environment
    nbh_idx, offsets = environment_provider.get_environment(atoms)

    # Get neighbors and neighbor mask
    inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))

    # Get cells
    inputs[Properties.cell] = torch.FloatTensor(cell)
    inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))

    # If requested get neighbor lists for triples
    if collect_triples:
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
        inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
        inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

        inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
            offset_idx_j.astype(np.int)
        )
        inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
            offset_idx_k.astype(np.int)
        )

    return inputs


class AtomsConverter:
    """
    Convert ASE atoms object to an input suitable for the SchNetPack
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        device (str): Device for computation (default='cpu')
    """

    def __init__(
        self,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        device=torch.device("cpu"),
    ):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples

        # Get device
        self.device = device

    def __call__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = _convert_atoms(atoms, self.environment_provider, self.collect_triples)

        # Calculate masks
        inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
        mask = inputs[Properties.neighbors] >= 0
        inputs[Properties.neighbor_mask] = mask.float()
        inputs[Properties.neighbors] = (
            inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
        )

        if self.collect_triples:
            mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
            mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
            mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
            inputs[Properties.neighbor_pairs_mask] = mask_triples.float()

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.unsqueeze(0).to(self.device)

        return inputs

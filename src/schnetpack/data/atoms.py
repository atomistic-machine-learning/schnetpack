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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, Iterable, Union

import torch
from ase import Atoms
from ase.db import connect
from torch.utils.data import Dataset

from schnetpack import Structure

logger = logging.getLogger(__name__)

__all__ = [
    "ASEAtomsData",
]


class AtomsDataError(Exception):
    pass


class AtomsDataFormat(Enum):
    ASE = "ASE"


class AtomsDataMixin(ABC):
    """
    Base mixin class for atomistic data. Use together with PyTorch Dataset or IterableDataset
    to implement concrete data formats.

    Args:
        load_properties: Set of properties to be loaded and returned.
            If None, all properties in the ASE dB will be returned.
        load_properties: If True, load structure properties.
    """

    def __init__(
        self,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
    ):
        self.load_properties = load_properties
        self.load_structure = load_structure

    @property
    @abstractmethod
    def available_properties(self) -> List[str]:
        """ Available properties in the dataset """
        pass

    @property
    def load_properties(self) -> List[str]:
        """ Properties to be loaded """
        return self._load_properties or self.available_properties

    @load_properties.setter
    def load_properties(self, val: List[str]):
        props = self.available_properties
        assert all(
            [p in props for p in val]
        ), "Not all given properties are available in the dataset!"
        self._load_properties = val

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """ Global metadata """
        pass

    @abstractmethod
    def update_metadata(self, **kwargs):
        pass


class ASEAtomsData(Dataset, AtomsDataMixin):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database.

    Args:
        datapath: Path to ASE DB.
        load_properties: Set of properties to be loaded and returned.
            If None, all properties in the ASE dB will be returned.
        load_properties: If True, load structure properties.
    """

    # TODO: add conversion for deprecated data

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
    ):
        self.datapath = datapath
        self.conn = None
        self._check_db()

        AtomsDataMixin.__init__(
            self,
            load_properties=load_properties,
            load_structure=load_structure,
        )

    def __len__(self) -> int:
        with connect(self.datapath) as conn:
            return conn.count()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with connect(self.datapath) as conn:
            props = self._get_properties(
                conn, self.load_properties, self.load_structure
            )
        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exists at {self.datapath}")

    def get_properties(
        self,
        indices: Union[int, Iterable[int]],
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        """
        Return property dictionary at given indices.

        Args:
            indices: data indices
            load_properties (sequence or None): subset of available properties to load

        Returns:
            properties (dict): dictionary with molecular properties

        """
        # use all available properties if nothing is specified
        if load_properties is None:
            load_properties = self.load_properties

        if load_properties is None:
            load_structure = self.load_structure

        if type(indices) is int:
            indices = [indices]

        # read from ase db
        properties = []
        with connect(self.datapath) as conn:
            for i in indices:
                properties.append(
                    self._get_properties(conn, i, load_properties, load_structure)
                )

        return properties

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        properties = {}
        properties["_idx"] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = row.data[pname]

        Z = row["numbers"]
        properties[Structure.n_atoms] = torch.tensor(Z.shape[0])

        if load_structure:
            properties[Structure.Z] = Z
            properties[Structure.position] = torch.tensor(row["positions"])
            properties[Structure.cell] = torch.tensor(row["cell"])
            properties[Structure.pbc] = torch.tensor(row["pbc"])

        return properties

    # Metadata

    @property
    def metadata(self):
        with connect(self.datapath) as conn:
            return conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        with connect(self.datapath) as conn:
            conn.metadata = val

    def update_metadata(self, **kwargs):
        assert all(
            key[0] != 0 for key in kwargs
        ), "Metadata keys starting with '_' are protected!"

        md = self.metadata
        md.update(kwargs)
        self._set_metadata(md)

    @property
    def available_properties(self) -> List[str]:
        md = self.metadata
        return md["_available_properties"]

    ## Creation

    @staticmethod
    def create(
        datapath: str,
        available_properties: List[str],
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
    ) -> "ASEAtomsData":
        """

        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_properties: If True, load structure properties.

        Returns:

        """
        if not datapath.endswith(".db"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        with connect(datapath) as conn:
            conn.metadata = {"_available_properties": available_properties}

        return ASEAtomsData(
            datapath, load_properties=load_properties, load_structure=load_structure
        )

    # add systems
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dict
                (using Structure.Z, Structure.R, Structure.cell, Structure.pbc)
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.datapath) as conn:
            self._add_system(conn, atoms, **properties)

    def add_systems(
        self,
        property_list: List[Dict[str:Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using Structure.Z, Structure.R, Structure.cell, Structure.pbc)
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        with connect(self.datapath) as conn:
            for at, prop in zip(atoms_list, property_list):
                self._add_system(conn, at, **prop)

    def _add_system(self, conn, atoms: Optional[Atoms] = None, **properties):
        """ Add systems to DB """
        if atoms is None:
            try:
                Z = properties[Structure.Z]
                R = properties[Structure.R]
                cell = properties[Structure.cell]
                pbc = properties[Structure.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys"
                ) from e

        data = {}
        # add available properties to database
        for pname in self.available_properties:
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data)

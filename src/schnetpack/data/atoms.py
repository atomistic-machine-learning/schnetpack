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
    "BaseAtomsData",
    "AtomsDataFormat",
    "resolve_format",
    "create_dataset",
    "load_dataset",
]


class AtomsDataFormat(Enum):
    ASE = "ase"


class AtomsDataError(Exception):
    pass


extension_map = {AtomsDataFormat.ASE: ".db"}


class BaseAtomsData(ABC):
    """
    Base mixin class for atomistic data. Use together with PyTorch Dataset or IterableDataset
    to implement concrete data formats.


    """

    def __init__(
        self,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[torch.nn.Module] = None,
    ):
        """

        Args:
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_properties: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
        """
        self.load_properties = load_properties
        self.load_structure = load_structure
        self.transforms = transforms

    @property
    @abstractmethod
    def available_properties(self) -> List[str]:
        """ Available properties in the dataset """
        pass

    @property
    @abstractmethod
    def units(self) -> Dict[str, str]:
        """ Available properties in the dataset """
        pass

    @property
    def load_properties(self) -> List[str]:
        """ Properties to be loaded """
        return self._load_properties or self.available_properties

    @load_properties.setter
    def load_properties(self, val: List[str]):
        if val is not None:
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

    @abstractmethod
    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        pass

    @staticmethod
    @abstractmethod
    def create(
        datapath: str, position_unit: str, property_unit_dict: Dict[str, str], **kwargs
    ) -> "BaseAtomsData":
        pass

    @abstractmethod
    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        pass

    @abstractmethod
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        pass


class ASEAtomsData(BaseAtomsData):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database.

    """

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_properties: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
        """
        self.datapath = datapath
        self.conn = None
        self._check_db()

        BaseAtomsData.__init__(
            self,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
        )

    def __len__(self) -> int:
        with connect(self.datapath) as conn:
            return conn.count()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with connect(self.datapath) as conn:
            props = self._get_properties(
                conn, idx, self.load_properties, self.load_structure
            )

        if self.transforms:
            props = self.transforms(props)

        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exists at {self.datapath}")

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
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

        if load_structure is None:
            load_structure = self.load_structure

        if indices is None:
            indices = range(len(self))
        elif type(indices) is int:
            indices = [indices]

        # read from ase db
        properties = []
        with connect(self.datapath) as conn:
            for i in indices:
                yield self._get_properties(
                    conn,
                    i,
                    load_properties=load_properties,
                    load_structure=load_structure,
                )

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[Structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = torch.tensor(row.data[pname].copy())

        Z = row["numbers"].copy()
        properties[Structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[Structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[Structure.position] = torch.tensor(row["positions"].copy())
            properties[Structure.cell] = torch.tensor(row["cell"].copy())
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
        return list(md["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        """ Dictionary of propterties to units """
        md = self.metadata
        return md["_property_unit_dict"]

    ## Creation

    @staticmethod
    def create(
        datapath: str, distance_unit: str, property_unit_dict: Dict[str, str], **kwargs
    ) -> "ASEAtomsData":
        """

        Args:
            datapath: Path to ASE DB.
            distance_unit: unit of atom positions and cell
            property_unit_dict: defines all properties with units of the dataset.
            kwargs: Pass arguments to init.

        Returns:
            newly created ASEAtomsData

        """
        if not datapath.endswith(".db"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        with connect(datapath) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_distance_unit": distance_unit,
            }

        return ASEAtomsData(datapath, **kwargs)

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
        property_list: List[Dict[str, Any]],
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
        for pname in conn.metadata["_property_unit_dict"].keys():
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data)


def create_dataset(
    datapath: str,
    format: AtomsDataFormat,
    distance_unit: str,
    property_unit_dict: Dict[str, str],
    **kwargs,
) -> BaseAtomsData:
    if format is AtomsDataFormat.ASE:
        dataset = ASEAtomsData.create(
            datapath=datapath,
            distance_unit=distance_unit,
            property_unit_dict=property_unit_dict,
            **kwargs,
        )
    else:
        raise AtomsDataError(f"Unknown format: {format}")
    return dataset


def load_dataset(datapath: str, format: AtomsDataFormat, **kwargs) -> Dataset:
    if format is AtomsDataFormat.ASE:
        dataset = ASEAtomsData(datapath=datapath, **kwargs)
    else:
        raise AtomsDataError(f"Unknown format: {format}")
    return dataset


def resolve_format(datapath: str, format: Optional[AtomsDataFormat]):
    file, suffix = os.path.splitext(datapath)
    if suffix == ".db":
        if format is None:
            format = AtomsDataFormat.ASE
        assert (
            format is AtomsDataFormat.ASE
        ), f"File extension {suffix} is not compatible with chosen format {format}"
    elif len(suffix) == 0 and format:
        datapath = datapath + extension_map[format]
    elif len(suffix) == 0 and format is None:
        raise AtomsDataError(
            "If format is not given, `datapath` needs a supported file extension!"
        )
    else:
        raise AtomsDataError(f"Unsupported file extension: {suffix}")
    return datapath, format

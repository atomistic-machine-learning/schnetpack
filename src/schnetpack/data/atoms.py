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
from typing import Optional, List, Dict, Any, Iterable, Union, Tuple

import torch
import copy
from ase import Atoms
from ase.db import connect
import pickle
import lmdb
from tqdm.auto import tqdm

import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.transform import Transform

logger = logging.getLogger(__name__)

__all__ = [
    "ASEAtomsData",
    "LMDBAtomsData",
    "BaseAtomsData",
    "AtomsDataFormat",
    "resolve_format",
    "create_dataset",
    "load_dataset",
]


class AtomsDataFormat(Enum):
    """Enumeration of data formats"""

    ASE = "ase"
    LMDB = "lmdb"


class AtomsDataError(Exception):
    pass


extension_map = {AtomsDataFormat.ASE: ".db", AtomsDataFormat.LMDB: ".lmdb"}


class BaseAtomsData(ABC):
    """
    Base mixin class for atomistic data. Use together with PyTorch Dataset or
    IterableDataset to implement concrete data formats.
    """

    def __init__(
        self,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
    ):
        """
        Args:
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing transforms (see schnetpack.data.transforms)
            subset: List of data indices.
        """
        self._transform_module = None
        self.load_properties = load_properties
        self.load_structure = load_structure
        self.transforms = transforms
        self.subset_idx = subset_idx

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value: Optional[List[Transform]]):
        self._transforms = []
        self._transform_module = None

        if value is not None:
            for tf in value:
                self._transforms.append(tf)
            self._transform_module = torch.nn.Sequential(*self._transforms)

    def subset(self, subset_idx: List[int]):
        assert (
            subset_idx is not None
        ), "Indices for creation of the subset need to be provided!"
        ds = copy.copy(self)
        if ds.subset_idx:
            ds.subset_idx = [ds.subset_idx[i] for i in subset_idx]
        else:
            ds.subset_idx = subset_idx
        return ds

    @property
    @abstractmethod
    def available_properties(self) -> List[str]:
        """Available properties in the dataset"""
        pass

    @property
    @abstractmethod
    def units(self) -> Dict[str, str]:
        """Property to unit dict"""
        pass

    @property
    def load_properties(self) -> List[str]:
        """Properties to be loaded"""
        if self._load_properties is None:
            return self.available_properties
        else:
            return self._load_properties

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
        """Global metadata"""
        pass

    @property
    @abstractmethod
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        """Single-atom reference values for properties"""
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
        datapath: str,
        position_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Dict[str, List[float]],
        **kwargs,
    ) -> "BaseAtomsData":
        pass

    @abstractmethod
    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
        key_value_list: Optional[List[Dict[str, Any]]] = None,
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
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        """
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
            subset_idx: List of data indices.
            units: property-> unit string dictionary that overwrites the native units
                of the dataset. Units are converted automatically during loading.
        """
        self.datapath = datapath
        self.conn = connect(self.datapath, use_lock_file=False)

        BaseAtomsData.__init__(
            self,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
        )

        self._check_db()

        # initialize units
        md = self.metadata
        if "_distance_unit" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a distance unit set. Please add units to the "
                + "dataset using `spkconvert`!"
            )
        if "_property_unit_dict" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a property units set. Please add units to the "
                + "dataset using `spkconvert`!"
            )

        if distance_unit:
            self.distance_conversion = spk.units.convert_units(
                md["_distance_unit"], distance_unit
            )
            self.distance_unit = distance_unit
        else:
            self.distance_conversion = 1.0
            self.distance_unit = md["_distance_unit"]

        self._units = md["_property_unit_dict"]
        self.conversions = {prop: 1.0 for prop in self._units}
        if property_units is not None:
            for prop, unit in property_units.items():
                self.conversions[prop] = spk.units.convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit

    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)

        return self.conn.count()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        props = self._get_properties(
            self.conn, idx, self.load_properties, self.load_structure
        )
        props = self._apply_transforms(props)

        return props

    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exist at {self.datapath}")

        if self.subset_idx:
            with connect(self.datapath, use_lock_file=False) as conn:
                n_structures = conn.count()

            assert max(self.subset_idx) < n_structures

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
            load_structure: load and return structure

        Returns:
            properties (dict): dictionary with molecular properties

        """
        if load_properties is None:
            load_properties = self.load_properties
        load_structure = load_structure or self.load_structure

        if self.subset_idx:
            if indices is None:
                indices = self.subset_idx
            elif type(indices) is int:
                indices = [self.subset_idx[indices]]
            else:
                indices = [self.subset_idx[i] for i in indices]
        else:
            if indices is None:
                indices = range(len(self))
            elif type(indices) is int:
                indices = [indices]

        # read from ase db
        for i in indices:
            yield self._get_properties(
                self.conn,
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
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(row.data[pname].copy()) * self.conversions[pname]
            )

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(row["positions"].copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(row["cell"][None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(row["pbc"])

        return properties

    # Metadata

    @property
    def metadata(self):
        return self.conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        self.conn.metadata = val

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
        """Dictionary of properties to units"""
        return self._units

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        md = self.metadata
        arefs = md["atomrefs"]
        arefs = {k: self.conversions[k] * torch.tensor(v) for k, v in arefs.items()}
        return arefs

    ## Creation

    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "ASEAtomsData":
        """

        Args:
            datapath: Path to ASE DB.
            distance_unit: unit of atom positions and cell
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            atomrefs: dictionary mapping properies (the keys) to lists of single-atom
                reference values of the property. This is especially useful for
                extensive properties such as the energy, where the single atom energies
                contribute a major part to the overall value.
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

        atomrefs = atomrefs or {}

        with connect(datapath) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_distance_unit": distance_unit,
                "atomrefs": atomrefs,
            }

        return ASEAtomsData(datapath, **kwargs)

    # add systems
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dict
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        self._add_system(self.conn, atoms, **properties)

    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
        key_value_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
            key_value_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        if key_value_list is None:
            key_value_list = [{}] * len(property_list)

        for at, prop, key_val in zip(atoms_list, property_list, key_value_list):
            self._add_system(
                self.conn,
                at,
                key_val,
                **prop,
            )

    def _add_system(
        self,
        conn,
        atoms: Optional[Atoms] = None,
        key_val: Optional[Dict[str, Any]] = None,
        **properties,
    ):
        """Add systems to DB"""
        if atoms is None:
            try:
                Z = properties[structure.Z]
                R = properties[structure.R]
                cell = properties[structure.cell]
                pbc = properties[structure.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys"
                ) from e

        # add available properties to database
        valid_props = set().union(
            conn.metadata["_property_unit_dict"].keys(),
            [structure.Z, structure.R, structure.cell, structure.pbc],
        )
        for prop in properties:
            if prop not in valid_props:
                logger.warning(
                    f"Property `{prop}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"AseAtomsData.create()."
                )
        for key in key_val:
            if key not in valid_props:
                logger.warning(
                    f"Property `{key}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"AseAtomsData.create()."
                )

        data = {}
        for pname in conn.metadata["_property_unit_dict"].keys():
            try:
                if pname in properties:
                    data[pname] = properties[pname]
                if pname in key_val:
                    data[pname] = key_val[pname]
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
    """
    Create a new atoms dataset.

    Args:
        datapath: file path
        format: atoms data format
        distance_unit: unit of atom positiona etc. as string
        property_unit_dict: dictionary that maps properties to units,
            e.g. {"energy": "kcal/mol"}
        **kwargs: arguments for passed to AtomsData init

    Returns:

    """
    if format is AtomsDataFormat.ASE:
        dataset = ASEAtomsData.create(
            datapath=datapath,
            distance_unit=distance_unit,
            property_unit_dict=property_unit_dict,
            **kwargs,
        )
    elif format is AtomsDataFormat.LMDB:
        dataset = LMDBAtomsData.create(
            datapath=datapath,
            distance_unit=distance_unit,
            property_unit_dict=property_unit_dict,
            **kwargs,
        )
    else:
        raise AtomsDataError(f"Unknown format: {format}")
    return dataset


def load_dataset(datapath: str, format: AtomsDataFormat, **kwargs) -> BaseAtomsData:
    """
    Load dataset.

    Args:
        datapath: file path
        format: atoms data format
        **kwargs: arguments for passed to AtomsData init

    """
    if format is AtomsDataFormat.ASE:
        dataset = ASEAtomsData(datapath=datapath, **kwargs)
    elif format is AtomsDataFormat.LMDB:
        dataset = LMDBAtomsData(datapath=datapath, **kwargs)
    else:
        raise AtomsDataError(f"Unknown format: {format}")
    return dataset


def resolve_format(
    datapath: str, format: Optional[AtomsDataFormat] = None
) -> Tuple[str, AtomsDataFormat]:
    """
    Extract data format from file suffix, check for consistency with (optional) given
    format, or append suffix to file path.

    Args:
        datapath: path to atoms data
        format: atoms data format

    """
    file, suffix = os.path.splitext(datapath)
    print(file, suffix)

    if suffix == extension_map[AtomsDataFormat.ASE]:
        if format is None:
            format = AtomsDataFormat.ASE
        assert (
            format is AtomsDataFormat.ASE
        ), f"File extension {suffix} is not compatible with chosen format {format}"
    elif suffix == extension_map[AtomsDataFormat.LMDB]:
        if format is None:
            format = AtomsDataFormat.LMDB
        assert (
            format is AtomsDataFormat.LMDB
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

class LMDBAtomsData(BaseAtomsData):
    """
    PyTorch dataset for atomistic data stored in a Lightning Memory-Mapped
    Database (LMDB) for fast I/O access during training.
    """

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        self.datapath = datapath
        self._writable_txn = None # Used only during creation
        self._current_idx = 0 # Used only during creation
        
        # Do NOT open the environment in the parent process.
        self.env = None
        
        # Load global metadata and total size
        try:
            with lmdb.open(
                datapath,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            ) as env:
                with env.begin(write=False) as txn:
                    metadata_bytes = txn.get(b"metadata")
                    if metadata_bytes is None:
                        raise AtomsDataError("LMDB metadata key 'metadata' not found.")
                    self._metadata = pickle.loads(metadata_bytes)
                    self._len = self._metadata.get("_len", 0)
        except Exception as e:
            raise AtomsDataError(f"Error opening LMDB to retrieve metadata: {e}")

        # Call base constructor
        BaseAtomsData.__init__(
            self,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
        )

        self._check_db()

        # Initialize units and conversions (similar to ASEAtomsData)
        md = self._metadata
        if "_distance_unit" not in md.keys() or "_property_unit_dict" not in md.keys():
             raise AtomsDataError(
                "Dataset does not have units set. Please add units to the "
                + "dataset using `spkconvert`!"
            )
        
        if distance_unit:
            self.distance_conversion = spk.units.convert_units(
                md["_distance_unit"], distance_unit
            )
            self.distance_unit = distance_unit
        else:
            self.distance_conversion = 1.0
            self.distance_unit = md["_distance_unit"]

        self._units = md["_property_unit_dict"]
        self.conversions = {prop: 1.0 for prop in self._units}
        if property_units is not None:
            for prop, unit in property_units.items():
                self.conversions[prop] = spk.units.convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit

    def _init_read_env(self):
        """Initializes the read-only LMDB environment if it is not already open."""
        if self.env is None:
            # Opening here ensures the environment is opened inside the worker process,
            # preventing the fork-related segmentation fault.
            self.env = lmdb.open(
                self.datapath,
                subdir=False,
                readonly=True,
                lock=False,  # Must be False for multiple simultaneous readers
                readahead=False,
                meminit=False,
            )
    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        self._init_read_env()

        props = self._get_properties(idx, self.load_properties, self.load_structure)
        props = self._apply_transforms(props)

        return props

    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"LMDB DB does not exist at {self.datapath}")

        if self.subset_idx:
            assert max(self.subset_idx) < self._len

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        if load_properties is None:
            load_properties = self.load_properties
        load_structure = load_structure or self.load_structure

        if self.subset_idx:
            if indices is None:
                indices = self.subset_idx
            elif isinstance(indices, int):
                indices = [self.subset_idx[indices]]
            else:
                indices = [self.subset_idx[i] for i in indices]
        else:
            if indices is None:
                indices = range(len(self))
            elif isinstance(indices, int):
                indices = [indices]
        
        for i in indices:
            yield self._get_properties(
                i,
                load_properties=load_properties,
                load_structure=load_structure,
            )

    def _get_properties(
        self, idx: int, load_properties: List[str], load_structure: bool
    ):
        """Helper function to load properties from LMDB for a single index."""
        with self.env.begin(write=False) as txn:
            data = txn.get(str(idx).encode("ascii"))
        
        if data is None:
            raise KeyError(f"Index {idx} not found in LMDB.")
            
        # Deserialization of the stored data structure
        structure_and_props = pickle.loads(data)
        
        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        
        # Molecular Properties
        for pname in load_properties:
            # Assumes properties are stored as numpy arrays or lists that can be converted to tensor
            properties[pname] = (
                torch.tensor(structure_and_props["properties"][pname].copy())
                * self.conversions[pname]
            )

        # Structure Properties
        atoms = structure_and_props["atoms"]
        Z = atoms.get_atomic_numbers()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])
        
        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(atoms.get_positions().copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(atoms.get_cell()[None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(atoms.get_pbc())
            
        return properties

    # Metadata Implementation (ReadOnly)

    @property
    def metadata(self):
        return self._metadata

    def _set_metadata(self, val: Dict[str, Any]):
        raise AtomsDataError("Cannot modify metadata on a read-only LMDB dataset instance.")

    def update_metadata(self, **kwargs):
        raise AtomsDataError("Cannot modify metadata on a read-only LMDB dataset instance.")

    @property
    def available_properties(self) -> List[str]:
        return list(self._metadata["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        return self._units

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        arefs = self._metadata["atomrefs"]
        arefs = {k: self.conversions[k] * torch.tensor(v) for k, v in arefs.items()}
        return arefs

    ## Creation (Requires external utility to commit the final database)

    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        map_size: int = 1099511627776, # Default 1TB, adjust as needed for large data
        **kwargs,
    ) -> "LMDBAtomsData":
        """
        Creates a new LMDB dataset file/directory and initializes it with metadata.
        Returns a temporary writable instance. Call `.close()` after adding all systems.
        """
        if os.path.exists(datapath):
            if os.path.isdir(datapath) or os.path.isfile(datapath):
                raise AtomsDataError(f"Dataset already exists: {datapath}")
                
        atomrefs = atomrefs or {}
        metadata = {
            "_property_unit_dict": property_unit_dict,
            "_distance_unit": distance_unit,
            "atomrefs": atomrefs,
            "_len": 0,
        }

        # Open LMDB Environment in write mode
        env = lmdb.open(
            datapath,
            subdir=False,
            map_size=map_size,
            max_dbs=1,
            create=True,

            readonly=False, 
            lock=True,
        )
        
        # Write initial metadata
        with env.begin(write=True) as txn:
            txn.put(b"metadata", pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))
            
        # Create a temporary instance and give it a persistent writable connection
        ds = LMDBAtomsData(datapath, **kwargs)
        ds.env.close()
        
        # Re-open with writable access and map_size for subsequent additions
        ds.env = lmdb.open(
            datapath,
            subdir=False,
            map_size=map_size,
            max_dbs=1,
            create=False,
            readonly=False, 
            lock=True,
        )
        ds._writable_txn = ds.env.begin(write=True) # Persistent write transaction
        ds._current_idx = 0
        
        return ds

    def close(self):
        """Commits the current transaction and closes the environment."""
        if hasattr(self, '_writable_txn') and self._writable_txn:
            # Final update of the dataset length
            self._metadata["_len"] = self._current_idx
            self._writable_txn.put(
                b"metadata", 
                pickle.dumps(self._metadata, protocol=pickle.HIGHEST_PROTOCOL)
            )
            self._writable_txn.commit()
            self._writable_txn = None
            
        if hasattr(self, 'env') and self.env:
            self.env.close()
            self.env = None

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()

    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        """
        Add atoms data to the LMDB dataset.
        """
        if not hasattr(self, '_writable_txn') or not self._writable_txn:
            raise AtomsDataError("LMDB dataset must be created/opened in write mode before adding systems.")
            
        current_idx = self._current_idx

        # Prepare ASE Atoms object
        if atoms is None:
            try:
                Z = properties[structure.Z]
                R = properties[structure.R]
                cell = properties[structure.cell]
                pbc = properties[structure.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys when `atoms` is None."
                ) from e
        
        # Extract properties to store
        data_to_store = {"atoms": atoms, "properties": {}}
        valid_props = set(self._metadata["_property_unit_dict"].keys())

        for pname in valid_props:
            if pname in properties:
                data_to_store["properties"][pname] = properties[pname]

        # Serialize and write to LMDB
        key = str(current_idx).encode("ascii")
        # Use a high-protocol pickle for efficiency
        value = pickle.dumps(data_to_store, protocol=pickle.HIGHEST_PROTOCOL)
        self._writable_txn.put(key, value)

        self._current_idx += 1
        
    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
        key_value_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add multiple systems to the dataset.
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        if key_value_list is None:
            key_value_list = [{}] * len(property_list)
        
        if not hasattr(self, '_writable_txn') or not self._writable_txn:
            raise AtomsDataError("LMDB dataset must be created/opened in write mode before adding systems.")

        for at, prop, key_val in tqdm(
            zip(atoms_list, property_list, key_value_list), 
            total=len(property_list), 
            desc="Adding systems to LMDB"
        ):
            # Combine properties from both lists for a single write
            combined_props = prop.copy()
            combined_props.update(key_val)
            self.add_system(at, **combined_props)
        
        # Update in-memory length after adding systems
        self._len = self._current_idx

# vim:ts=4 sw=4 et

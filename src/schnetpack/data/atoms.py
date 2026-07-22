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

import copy
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterable, Union

import fasteners
import torch
from ase import Atoms
from ase.db import connect

import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.transform.base import Transform

logger = logging.getLogger(__name__)

__all__ = ["ASEAtomsData", "DownloadableASEAtomsData", "AtomsDataError"]


class AtomsDataError(Exception):
    pass


class ASEAtomsData(torch.utils.data.Dataset):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database.
    """

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
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
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: List of data indices.
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """
        self.datapath = datapath
        self.subset_idx = subset_idx
        # ASEAtomsData is a pure reader of existing DBs — a missing file is
        # always an error here. Datasets that can fetch their own data
        # subclass DownloadableASEAtomsData instead.
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exist at {self.datapath}")

        self._check_db()
        self.conn = connect(self.datapath, use_lock_file=False)

        self.transforms = list(transforms or [])
        self.train_transforms = list(train_transforms) if train_transforms else None
        self.val_transforms = list(val_transforms) if val_transforms else None
        self.test_transforms = list(test_transforms) if test_transforms else None
        self.split = None

        self._load_properties: Optional[List[str]] = None
        self.load_structure = load_structure

        # units from metadata
        md = self.metadata
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

        # apply unit overrides on load only
        if property_units is not None:
            for prop, unit in property_units.items():
                self.conversions[prop] = spk.units.convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit

        # now validate load_properties against available_properties
        self.load_properties = load_properties

    # ---------- merged ASEAtomsData bits ----------

    def subset(self, subset_idx: List[int], split: Optional[str] = None):
        if subset_idx is None:
            raise ValueError("subset_idx must be provided.")
        ds = copy.copy(self)
        if ds.subset_idx is not None:
            ds.subset_idx = [ds.subset_idx[i] for i in subset_idx]
        else:
            ds.subset_idx = subset_idx
        ds.split = split
        return ds

    @property
    def load_properties(self) -> List[str]:
        if self._load_properties is None:
            return self.available_properties
        return self._load_properties

    @load_properties.setter
    def load_properties(self, val: Optional[List[str]]):
        if val is not None:
            props = self.available_properties
            missing = [p for p in val if p not in props]
            if missing:
                raise AtomsDataError(f"Properties not available in dataset: {missing}")
        self._load_properties = val

    # ---------- core dataset API ----------

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
        return self._apply_transforms(props)

    def get_split_transforms(self) -> List[Transform]:
        if self.split == "train" and self.train_transforms is not None:
            return self.train_transforms
        if self.split == "val" and self.val_transforms is not None:
            return self.val_transforms
        if self.split == "test" and self.test_transforms is not None:
            return self.test_transforms
        return self.transforms

    def initialize_transforms(self, stats=None) -> None:
        for tf in self.get_split_transforms():
            if hasattr(tf, "initialize"):
                tf.initialize(stats)

    def _apply_transforms(
        self, props: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for tf in self.get_split_transforms():
            props = tf(props)
        return props

    def _check_db(self) -> None:
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exist at {self.datapath}")

        with connect(self.datapath, use_lock_file=False) as conn:
            n_structures = conn.count()
            md = conn.metadata

        # An empty DB is legal: it is the starting point of the `create()` +
        # `add_systems()` workflow used for custom datasets.

        if "_distance_unit" not in md:
            raise AtomsDataError(
                "Dataset does not have a distance unit set. Please add units to the dataset."
            )

        if "_property_unit_dict" not in md:
            raise AtomsDataError(
                "Dataset does not have property units set. Please add units to the dataset."
            )

        if self.subset_idx is not None:
            if max(self.subset_idx) >= n_structures:
                raise AtomsDataError("subset_idx contains out-of-range indices")

    # ---------- metadata / units -----------

    @property
    def metadata(self) -> Dict[str, Any]:
        with connect(self.datapath, use_lock_file=False) as conn:
            return conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        with connect(self.datapath, use_lock_file=False) as conn:
            conn.metadata = val

    def update_metadata(self, **kwargs):
        if not all(k and k[0] != "_" for k in kwargs):
            raise AtomsDataError("Metadata keys starting with '_' are protected!")
        md = self.metadata
        md.update(kwargs)
        self._set_metadata(md)

    @property
    def available_properties(self) -> List[str]:
        md = self.metadata
        return list(md["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        return self._units

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        md = self.metadata
        arefs = md.get("atomrefs", {})
        return {k: self.conversions[k] * torch.tensor(v) for k, v in arefs.items()}

    # ---------- iteration ----------

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: Optional[List[str]] = None,
        load_structure: Optional[bool] = None,
        load_metadata: bool = False,
    ):
        """
        Return property dictionary at given indices.

        Args:
            indices: data indices
            load_properties (sequence or None): subset of available properties to load
            load_structure: load and return structure
            load_metadata: load and return metadata

        Returns:
            properties (dict): dictionary with molecular properties

        """
        if load_properties is None:
            load_properties = self.load_properties
        if load_structure is None:
            load_structure = self.load_structure

        if self.subset_idx is not None:
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
                self.conn,
                i,
                load_properties=load_properties,
                load_structure=load_structure,
                load_metadata=load_metadata,
            )

    def _get_properties(
        self,
        conn,
        idx: int,
        load_properties: List[str],
        load_structure: bool,
        load_metadata: bool = False,
    ):
        """
        Load properties of a single system from the ASE database.

        Args:
            conn: ASE database connection.
            idx: Zero-based system index.
            load_properties: Properties to load.
            load_structure: Whether to load structural information.
            load_metadata: Whether to load metadata.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the requested properties.
        """
        row = conn.get(idx + 1)
        # TODO: can the copies be avoided?
        properties: Dict[str, torch.Tensor] = {}
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

        if load_metadata:
            properties["metadata"] = row.key_value_pairs

        return properties

    # ---------- creation / writing ----------

    @classmethod
    def create(
        cls,
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "ASEAtomsData":
        """
        Create a new, empty ASE database and return a dataset for it.

        This is the public creation API for custom datasets:
        ``ASEAtomsData.create(...)`` followed by ``add_systems(...)``.

        Args:
            datapath: Path to the new ASE DB (must end in ``.db``).
            distance_unit: unit of atom positions and cell.
            property_unit_dict: available properties of the dataset mapped to
                their units. If a property is unit-less, pass "arb. unit" or
                `None`.
            atomrefs: dictionary mapping properties to lists of single-atom
                reference values.
            kwargs: passed on to the constructor.

        Returns:
            newly created dataset instance
        """
        cls._write_empty_db(datapath, distance_unit, property_unit_dict, atomrefs)
        return cls(datapath, **kwargs)

    @staticmethod
    def _write_empty_db(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        if not datapath.endswith(".db"):
            raise AtomsDataError("Invalid datapath! Add '.db' extension.")
        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")
        if property_unit_dict is None:
            raise AtomsDataError("property_unit_dict must be provided.")
        if distance_unit is None:
            raise AtomsDataError("distance_unit must be provided.")

        os.makedirs(os.path.dirname(datapath) or ".", exist_ok=True)

        with connect(datapath) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_distance_unit": distance_unit,
                "atomrefs": atomrefs or {},
            }

    def add_system(
        self,
        atoms: Optional[Atoms] = None,
        atoms_metadata: Optional[Dict[str, Any]] = None,
        **properties,
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dict
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            atoms_metadata: Metadata of the atoms object as key-value pairs.
                Metadata can not be used as a training property, but can be used for splitting
                strategies (e.g. material_id, timestamp, ...).
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        self._add_system(atoms, atoms_metadata, **properties)

    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
        atoms_metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add atoms data to the dataset.

        Args:
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            atoms_metadata_list: Metadata of the atoms objects as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Metadata can not be used as a training property, but can be used for splitting
                strategies (e.g. material_id, timestamp, ...).
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)
        if atoms_metadata_list is None:
            atoms_metadata_list = [{}] * len(property_list)

        for atoms, prop, atoms_metadata in zip(
            atoms_list, property_list, atoms_metadata_list
        ):
            self._add_system(atoms, atoms_metadata, **prop)

    def _add_system(
        self,
        atoms: Optional[Atoms] = None,
        atoms_metadata: Optional[Dict[str, Any]] = None,
        **properties,
    ):
        """
        Add systems to DB.
        """
        if atoms is None:
            try:
                Z = properties[structure.Z]
                R = properties[structure.R]
                cell = properties[structure.cell]
                pbc = properties[structure.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError("Missing structure keys in properties") from e

        if atoms_metadata is None:
            atoms_metadata = {}

        with connect(self.datapath, use_lock_file=False) as conn:
            prop_keys = conn.metadata["_property_unit_dict"].keys()

            valid_props = set(prop_keys).union(
                {structure.Z, structure.R, structure.cell, structure.pbc}
            )
            for pname in properties:
                if pname not in valid_props:
                    logger.warning(
                        f"Property `{pname}` is not defined for this dataset and will be ignored."
                    )

            data = {}
            for pname in prop_keys:
                if pname not in properties:
                    raise AtomsDataError("Required property missing: " + pname)
                data[pname] = properties[pname]

            conn.write(atoms, data=data, key_value_pairs=atoms_metadata)


class DownloadableASEAtomsData(ASEAtomsData, ABC):
    """
    Base class for datasets that can download and build their own ASE database.

    On instantiation, if `datapath` does not exist (and `download=True`), an
    empty DB is created from the subclass's native units and `download()` is
    called to fill it. Subclasses must set `self.property_units` (native
    property unit dict) and `self.distance_unit` BEFORE calling
    `super().__init__()`, and must implement `download()`.
    """

    def __init__(self, datapath: str, download: bool = True, **kwargs):
        """
        Args:
            datapath: Path to ASE DB.
            download: If True (default), download the dataset if `datapath`
                does not exist. If False, a missing `datapath` raises instead.
            **kwargs: arguments passed on to the ASEAtomsData constructor.
        """
        self.datapath = datapath
        if not os.path.exists(self.datapath):
            if not download:
                raise AtomsDataError(
                    f"ASE DB does not exist at {self.datapath}. "
                    "Pass download=True to download the dataset."
                )

            # Downloads happen in __init__, which every DDP rank executes.
            # Serialize concurrent creators with an inter-process lock; late
            # ranks find the finished DB and skip the download.
            os.makedirs(os.path.dirname(self.datapath) or ".", exist_ok=True)
            lock = fasteners.InterProcessLock(self.datapath + ".lock")
            with lock:
                if not os.path.exists(self.datapath):
                    # Write the empty DB from the subclass's native units,
                    # then let download() fill it.
                    self._write_empty_db(
                        self.datapath, self.distance_unit, self.property_units
                    )
                    try:
                        self.download()
                    except BaseException:
                        # A failed download must not leave a 0-row DB behind —
                        # it would permanently block retries, since the next
                        # run sees the file and skips the download.
                        if os.path.exists(self.datapath):
                            os.remove(self.datapath)
                        raise
                    # Catch download() implementations that return without
                    # writing anything.
                    with connect(self.datapath, use_lock_file=False) as conn:
                        if conn.count() == 0:
                            os.remove(self.datapath)
                            raise AtomsDataError(
                                f"download() of {self.__class__.__name__} "
                                f"produced an empty database at {self.datapath}"
                            )

        super().__init__(datapath=datapath, **kwargs)

    @abstractmethod
    def download(self):
        """Fill the freshly created, empty DB at `self.datapath`."""
        ...

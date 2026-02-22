import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
from urllib import request as request

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz
from tqdm import tqdm

import torch
import schnetpack.properties as structure
from schnetpack.data import (
    AtomsDataFormat,
    AtomsDataModuleError,
    BaseAtomsData,
    create_dataset,
    load_dataset,
)

__all__ = ["QM9"]


class QM9:
    """QM9 benchmark database for organic molecules (dataset-only).

    This class:
      - is a dataset wrapper (no Lightning DataModule inheritance)
      - can download + build the dataset via prepare()
      - forwards the BaseAtomsData API to an underlying dataset instance
    """

    base_urls = [
        "https://ndownloader.figshare.com/files/",
        "https://springernature.figshare.com/ndownloader/files/",
    ]
    file_ids = {
        "data": "3195389",
        "atomrefs": "3195395",
        "uncharacterized": "3195404",
    }

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"

    def __init__(
        self,
        datapath: str,
        format: AtomsDataFormat = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        remove_uncharacterized: bool = False,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset DB (e.g. qm9.db)
            format: dataset format (ASE by default)
            load_properties: subset of properties to load
            remove_uncharacterized: if True, exclude uncharacterized molecules
            property_units: optional unit overrides on load (passed to load_dataset/create_dataset)
            distance_unit: optional distance unit override on load (passed to load_dataset/create_dataset)
            transforms: optional default transforms (typically set by your DataModule per split)
            **kwargs: reserved for forward compatibility
        """
        self.datapath = datapath
        self.format = format
        self.load_properties = load_properties
        self.remove_uncharacterized = remove_uncharacterized
        self.property_units = property_units
        self.distance_unit = distance_unit
        self._kwargs = kwargs

        self._dataset: Optional[BaseAtomsData] = None
        self.transforms = transforms or []

    # -------------------------
    # Dataset forwarding helpers
    # -------------------------
    def _ensure_loaded(self) -> None:
        if self._dataset is None:
            # Lazy-load if it already exists; otherwise require user to call prepare()
            if not os.path.exists(self.datapath):
                raise AtomsDataModuleError(
                    f"QM9 dataset not found at {self.datapath}. Call dataset.prepare() first."
                )
            self._dataset = load_dataset(
                self.datapath,
                self.format,
                load_properties=self.load_properties,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
            )
            # attach any transforms set before first load
            self._dataset.transforms = self.transforms

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._dataset)

    def __getitem__(self, idx: int):
        self._ensure_loaded()
        return self._dataset[idx]

    def subset(self, indices):
        """
        Forward subset() to underlying dataset.
        Returns a BaseAtomsData-like object (whatever the backend returns).
        """
        self._ensure_loaded()
        sub = self._dataset.subset(indices)
        # Ensure transforms are carried over if caller expects it
        # (DataModuleV2 will set per-split transforms anyway)
        if getattr(sub, "transforms", None) is None:
            sub.transforms = []
        return sub

    # Common attributes used elsewhere in SchNetPack
    @property
    def available_properties(self):
        self._ensure_loaded()
        return self._dataset.available_properties

    @property
    def atomrefs(self):
        self._ensure_loaded()
        return getattr(self._dataset, "atomrefs", None)

    @property
    def metadata(self):
        self._ensure_loaded()
        return getattr(self._dataset, "metadata", None)

    @property
    def distance_unit_internal(self):
        self._ensure_loaded()
        return getattr(self._dataset, "distance_unit", None)

    @property
    def property_unit_dict(self):
        self._ensure_loaded()
        return getattr(self._dataset, "property_unit_dict", None)

    # -------------------------
    # Download / build pipeline
    # -------------------------
    def _download_file(self, file_id: str, destination: str) -> None:
        for base_url in self.base_urls:
            url = f"{base_url}{file_id}"
            try:
                request.urlretrieve(url, destination)
                return
            except Exception:
                logging.warning(f"Could not download from {url}, trying next source...")
        raise AtomsDataModuleError(
            f"Could not download file with id {file_id} from any source."
        )

    def prepare_data(self) -> None:
        """
        Download + build the dataset if missing. If it already exists, verify
        the uncharacterized setting is consistent.

        After prepare(), the dataset is loaded and ready to use.
        """
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                QM9.A: "GHz",
                QM9.B: "GHz",
                QM9.C: "GHz",
                QM9.mu: "Debye",
                QM9.alpha: "a0 a0 a0",
                QM9.homo: "Ha",
                QM9.lumo: "Ha",
                QM9.gap: "Ha",
                QM9.r2: "a0 a0",
                QM9.zpve: "Ha",
                QM9.U0: "Ha",
                QM9.U: "Ha",
                QM9.H: "Ha",
                QM9.G: "Ha",
                QM9.Cv: "cal/mol/K",
            }

            tmpdir = tempfile.mkdtemp("qm9")
            try:
                atomrefs = self._download_atomrefs(tmpdir)

                dataset = create_dataset(
                    datapath=self.datapath,
                    format=self.format,
                    distance_unit=self.distance_unit or "Ang",
                    property_unit_dict=(
                        property_unit_dict
                        if self.property_units is None
                        else self.property_units
                    ),
                    atomrefs=atomrefs,
                )

                if self.remove_uncharacterized:
                    uncharacterized = self._download_uncharacterized(tmpdir)
                else:
                    uncharacterized = None

                self._download_data(tmpdir, dataset, uncharacterized=uncharacterized)

            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        else:
            # validate uncharacterized constraint against dataset size
            dataset = load_dataset(
                self.datapath,
                self.format,
                load_properties=self.load_properties,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
            )
            if self.remove_uncharacterized and len(dataset) == 133885:
                raise AtomsDataModuleError(
                    "The dataset at the chosen location contains the uncharacterized 3054 molecules. "
                    "Choose a different location to reload the data or set `remove_uncharacterized=False`."
                )
            if (not self.remove_uncharacterized) and len(dataset) < 133885:
                raise AtomsDataModuleError(
                    "The dataset at the chosen location does NOT contain the uncharacterized 3054 molecules. "
                    "Choose a different location to reload the data or set `remove_uncharacterized=True`."
                )

        # Load after prepare so wrapper is ready
        self._dataset = load_dataset(
            self.datapath,
            self.format,
            load_properties=self.load_properties,
            property_units=self.property_units,
            distance_unit=self.distance_unit,
        )
        self._dataset.transforms = self.transforms

    # # keep Lightning naming for convenience if any code still calls it
    # def prepare_data(self) -> None:
    #     self.prepare()

    def _download_uncharacterized(self, tmpdir: str) -> List[int]:
        logging.info("Downloading list of uncharacterized molecules...")
        tmp_path = os.path.join(tmpdir, "uncharacterized.txt")
        self._download_file(self.file_ids["uncharacterized"], tmp_path)
        logging.info("Done.")

        uncharacterized = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_atomrefs(self, tmpdir: str) -> Dict[str, List[float]]:
        logging.info("Downloading GDB-9 atom references...")
        tmp_path = os.path.join(tmpdir, "atomrefs.txt")
        self._download_file(self.file_ids["atomrefs"], tmp_path)
        logging.info("Done.")

        props = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]
        atref = {p: np.zeros((100,)) for p in props}
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                for i, p in enumerate(props):
                    atref[p][z] = float(l.split()[i + 1])
        return {k: v.tolist() for k, v in atref.items()}

    def _download_data(
        self,
        tmpdir: str,
        dataset: BaseAtomsData,
        uncharacterized: Optional[List[int]],
    ) -> None:
        logging.info("Downloading GDB-9 data...")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        self._download_file(self.file_ids["data"], tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub(r"\D", "", x)), x)
        )

        property_list = []
        irange = np.arange(len(ordered_files), dtype=int)
        if uncharacterized is not None:
            irange = np.setdiff1d(irange, np.array(uncharacterized, dtype=int) - 1)

        for i in tqdm(irange):
            xyzfile = os.path.join(raw_path, ordered_files[i])
            properties = {}

            tmp = io.StringIO()
            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(dataset.available_properties, l):
                    properties[pn] = np.array([float(p)])
                for line in lines:
                    tmp.write(line.replace("*^", "e"))

            tmp.seek(0)
            ats: Atoms = list(read_xyz(tmp, 0))[0]
            properties[structure.Z] = ats.numbers
            properties[structure.R] = ats.positions
            properties[structure.cell] = ats.cell
            properties[structure.pbc] = ats.pbc
            property_list.append(properties)

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")

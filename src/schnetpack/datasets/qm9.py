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
from ase.db import connect

from tqdm import tqdm

import schnetpack.properties as structure
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform


__all__ = ["QM9"]


class QM9(ASEAtomsData):
    """
    QM9 benchmark database for organic molecules.
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
        remove_uncharacterized: bool = False,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            remove_uncharacterized: do not include uncharacterized molecules.
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """
        self.remove_uncharacterized = remove_uncharacterized
        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        return {
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

    def _check_db(self) -> None:
        super()._check_db()
        with connect(self.datapath, use_lock_file=False) as conn:
            data_count = conn.count()

        if self.remove_uncharacterized and data_count == 133885:
            raise AtomsDataError(
                "The dataset at the chosen location contains the uncharacterized 3054 molecules. "
                "Choose a different location to reload the data or set "
                "`remove_uncharacterized=False`."
            )

        if (not self.remove_uncharacterized) and data_count < 133885:
            raise AtomsDataError(
                "The dataset at the chosen location does NOT contain the uncharacterized 3054 molecules. "
                "Choose a different location to reload the data or set "
                "`remove_uncharacterized=True`."
            )

    def download(self) -> None:
        tmpdir = tempfile.mkdtemp("qm9")

        atomrefs = self._download_atomrefs(tmpdir)
        md = self.metadata
        md["atomrefs"] = atomrefs
        self._set_metadata(md)

        if self.remove_uncharacterized:
            uncharacterized = self._download_uncharacterized(tmpdir)
        else:
            uncharacterized = None

        self._download_data(tmpdir, uncharacterized)

        shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_file(self, file_id: str, destination: str) -> None:
        for base_url in self.base_urls:
            try:
                request.urlretrieve(f"{base_url}{file_id}", destination)
                return
            except Exception:
                continue
        raise AtomsDataError(
            f"Could not download file with id {file_id} from any source."
        )

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
        uncharacterized: Optional[List[int]],
    ) -> None:

        logging.info("Downloading GDB-9 data...")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        self._download_file(self.file_ids["data"], tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(raw_path)
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub(r"\D", "", x)), x)
        )

        property_list = []
        indices = np.arange(len(ordered_files), dtype=int)

        if uncharacterized is not None:
            indices = np.setdiff1d(indices, np.array(uncharacterized, dtype=int) - 1)

        for i in tqdm(indices):
            xyzfile = os.path.join(raw_path, ordered_files[i])
            properties = {}

            tmp = io.StringIO()
            with open(xyzfile, "r") as f:
                lines = f.readlines()
                values = lines[1].split()[2:]

                for pname, value in zip(self.available_properties, values):
                    properties[pname] = np.array([float(value)])

                for line in lines:
                    tmp.write(line.replace("*^", "e"))

            tmp.seek(0)
            atoms: Atoms = list(read_xyz(tmp, 0))[0]

            properties[structure.Z] = atoms.numbers
            properties[structure.R] = atoms.positions
            properties[structure.cell] = atoms.cell
            properties[structure.pbc] = atoms.pbc

            property_list.append(properties)

        logging.info("Write atoms to db...")
        self.add_systems(property_list=property_list)
        logging.info("Done.")

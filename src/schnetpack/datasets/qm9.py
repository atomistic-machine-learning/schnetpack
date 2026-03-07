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

import schnetpack.properties as structure
from schnetpack.data import AtomsDataFormat
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError

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
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        remove_uncharacterized: bool = False,
        load_properties: Optional[List[str]] = None,
        transforms=None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        self.remove_uncharacterized = remove_uncharacterized
        self.format = format

        self.download(
            datapath=datapath,
            distance_unit=distance_unit or "Ang",
        )

        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        # IMPORTANT: full native QM9 schema, stored in DB metadata
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

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        """
        Make sure the QM9 database exists.

        If the DB already exists, validate consistency with the
        remove_uncharacterized setting.
        """
        if os.path.exists(datapath):
            dataset = ASEAtomsData(datapath=datapath, load_structure=False)

            if self.remove_uncharacterized and len(dataset) == 133885:
                raise AtomsDataError(
                    "The dataset at the chosen location contains the uncharacterized 3054 molecules. "
                    "Choose a different location to reload the data or set "
                    "`remove_uncharacterized=False`."
                )

            if (not self.remove_uncharacterized) and len(dataset) < 133885:
                raise AtomsDataError(
                    "The dataset at the chosen location does NOT contain the uncharacterized 3054 molecules. "
                    "Choose a different location to reload the data or set "
                    "`remove_uncharacterized=True`."
                )
            return

        tmpdir = tempfile.mkdtemp("qm9")
        try:
            atomrefs = self._download_atomrefs(tmpdir)

            dataset = ASEAtomsData.create(
                datapath=datapath,
                distance_unit=distance_unit,
                property_unit_dict=self._native_property_units(),
                atomrefs=atomrefs,
            )

            if self.remove_uncharacterized:
                uncharacterized = self._download_uncharacterized(tmpdir)
            else:
                uncharacterized = None

            self._download_data(tmpdir, dataset, uncharacterized)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _download_file(self, file_id: str, destination: str) -> None:
        for base_url in self.base_urls:
            url = f"{base_url}{file_id}"
            try:
                request.urlretrieve(url, destination)
                return
            except Exception:
                logging.warning(f"Could not download from {url}, trying next source...")

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
        dataset: ASEAtomsData,
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

                for pname, value in zip(dataset.available_properties, values):
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
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")

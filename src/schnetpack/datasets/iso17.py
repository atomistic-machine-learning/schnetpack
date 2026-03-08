import logging
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
from urllib import request as request
from urllib.error import HTTPError, URLError

import torch
import numpy as np
from ase.db import connect
from tqdm import tqdm

from schnetpack.data import AtomsDataFormat
from schnetpack.data.atoms import ASEAtomsData, AtomsDataError

__all__ = ["ISO17"]


class ISO17(ASEAtomsData):
    """
    ISO17 benchmark dataset for molecular dynamics of C7O2H10 isomers
    containing molecular forces.

    References:

    .. [#iso17] http://quantum-machine.org/datasets/
    """

    energy = "total_energy"
    forces = "atomic_forces"

    existing_folds = [
        "reference",
        "reference_eq",
        "test_within",
        "test_other",
        "test_eq",
    ]

    def __init__(
        self,
        datapath: str,
        fold: str,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            fold: select a specific dataset of iso17
            format: dataset format
            load_properties: subset of properties to load
            transforms: Transform applied to each system separately before batching
            subset_idx: indices of the subset to load
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            **kwargs: additional keyword arguments.
        """
        if fold not in self.existing_folds:
            raise AtomsDataError(f"Fold {fold} does not exist.")

        self.root_path = datapath
        self.fold = fold
        self.format = format

        dbpath = os.path.join(datapath, "iso17", fold + ".db")

        self.download(datapath=dbpath, distance_unit=distance_unit or "Ang")

        super().__init__(
            datapath=dbpath,
            load_properties=load_properties,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

    @staticmethod
    def _native_property_units() -> Dict[str, str]:
        return {
            ISO17.energy: "eV",
            ISO17.forces: "eV/Ang",
        }

    def download(self, datapath: str, distance_unit: str = "Ang") -> None:
        """
        Ensure the ISO17 DB for the selected fold exists and has proper metadata.
        """
        if os.path.exists(datapath):
            _ = ASEAtomsData(datapath, load_structure=False)
            return

        self._download_data()

    def _download_data(self) -> None:
        logging.info("Downloading ISO17 database...")
        tmpdir = tempfile.mkdtemp("iso17")

        try:
            tarpath = os.path.join(tmpdir, "iso17.tar.gz")
            url = "http://www.quantum-machine.org/datasets/iso17.tar.gz"

            try:
                request.urlretrieve(url, tarpath)
            except HTTPError as e:
                raise AtomsDataError(
                    f"HTTP Error {e.code} while downloading {url}"
                ) from e
            except URLError as e:
                raise AtomsDataError(
                    f"URL Error {e.reason} while downloading {url}"
                ) from e

            with tarfile.open(tarpath) as tar:
                tar.extractall(self.root_path)

            # update metadata + convert energy into row.data for every fold
            for fold in self.existing_folds:
                dbpath = os.path.join(self.root_path, "iso17", fold + ".db")
                tmp_dbpath = os.path.join(tmpdir, f"{fold}_tmp.db")

                with connect(dbpath) as conn:
                    with connect(tmp_dbpath) as tmp_conn:
                        tmp_conn.metadata = {
                            "_property_unit_dict": self._native_property_units(),
                            "_distance_unit": "Ang",
                            "atomrefs": {},
                        }

                        for idx in tqdm(
                            range(len(conn)),
                            desc=f"parsing database file {dbpath}",
                        ):
                            atmsrw = conn.get(idx + 1)
                            data = atmsrw.data
                            data[self.forces] = np.array(data[self.forces])
                            data[self.energy] = np.array([atmsrw.total_energy])
                            tmp_conn.write(atmsrw.toatoms(), data=data)

                os.remove(dbpath)
                os.rename(tmp_dbpath, dbpath)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

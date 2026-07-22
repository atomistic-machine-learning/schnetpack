import logging
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
from urllib import request as request
from urllib.error import HTTPError, URLError

import numpy as np
from ase.db import connect
from tqdm import tqdm

from schnetpack.transform.base import Transform
from schnetpack.data.atoms import DownloadableASEAtomsData, AtomsDataError

__all__ = ["ISO17"]


class ISO17(DownloadableASEAtomsData):
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
            fold: select a specific dataset of iso17
            load_properties: subset of properties to load
            transforms: transform applied to each system separately before batching
            train_transforms: overrides transform_fn for training
            val_transforms: overrides transform_fn for validation
            test_transforms: overrides transform_fn for testing
            subset_idx: indices of the subset to load
            property_units: dictionary from property to corresponding unit as a string (eV, kcal/mol, ...)
            distance_unit: unit of the atom positions and cell as a string (Ang, Bohr, ...)
        """
        if fold not in self.existing_folds:
            raise AtomsDataError(f"Fold {fold} does not exist.")

        self.root_path = datapath
        self.fold = fold
        self.distance_unit = "Ang"
        self.property_units = self._native_property_units()

        dbpath = os.path.join(datapath, "iso17", fold + ".db")

        super().__init__(
            datapath=dbpath,
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
            ISO17.energy: "eV",
            ISO17.forces: "eV/Ang",
        }

    def download(self) -> None:
        logging.info("Downloading ISO17 database...")
        tmpdir = tempfile.mkdtemp("iso17")
        tarpath = os.path.join(tmpdir, "iso17.tar.gz")
        url = "http://www.quantum-machine.org/datasets/iso17.tar.gz"

        try:
            request.urlretrieve(url, tarpath)

        except HTTPError as e:
            raise AtomsDataError(f"HTTP Error {e.code} while downloading {url}") from e

        except URLError as e:
            raise AtomsDataError(f"URL Error {e.reason} while downloading {url}") from e

        with tarfile.open(tarpath) as tar:
            tar.extractall(self.root_path)

        # update metadata + convert energy into row.data for every fold
        for fold in self.existing_folds:
            dbpath = os.path.join(self.root_path, "iso17", fold + ".db")
            tmp_dbpath = os.path.join(tmpdir, f"{fold}_tmp.db")

            with connect(dbpath, use_lock_file=False) as conn:
                with connect(tmp_dbpath, use_lock_file=False) as tmp_conn:
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

        shutil.rmtree(tmpdir, ignore_errors=True)

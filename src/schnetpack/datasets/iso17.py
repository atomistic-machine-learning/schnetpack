import logging
import os
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError

import numpy as np
import torch
from ase.db import connect
from tqdm import tqdm

from schnetpack.data import (
    AtomsDataFormat,
    AtomsDataModule,
    load_dataset,
)

__all__ = ["ISO17"]


class ISO17(AtomsDataModule):
    """
    ISO17 benchmark data set for molecular dynamics of C7O2H10 isomers
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

    # properties
    def __init__(
        self,
        datapath: str,
        fold: str,
        batch_size: int,
        num_train: int | None = None,
        num_val: int | None = None,
        num_test: int | None = None,
        split_file: str | None = "split.npz",
        format: AtomsDataFormat | None = AtomsDataFormat.ASE,
        load_properties: list[str] | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        transforms: list[torch.nn.Module] | None = None,
        train_transforms: list[torch.nn.Module] | None = None,
        val_transforms: list[torch.nn.Module] | None = None,
        test_transforms: list[torch.nn.Module] | None = None,
        num_workers: int = 2,
        num_val_workers: int | None = None,
        num_test_workers: int | None = None,
        property_units: dict[str, str] | None = None,
        distance_unit: str | None = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
            fold: select a specific dataset of iso17
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
        """
        if fold not in self.existing_folds:
            raise ValueError(f"Fold {fold:s} does not exist")

        self.path = datapath
        self.fold = fold
        dbpath = os.path.join(datapath, "iso17", fold + ".db")

        super().__init__(
            datapath=dbpath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs,
        )

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            self._download_data()
        else:
            load_dataset(self.datapath, self.format)

    def _download_data(self):
        logging.info("Downloading ISO17 database...")
        tmpdir = tempfile.mkdtemp("iso17")
        tarpath = os.path.join(tmpdir, "iso17.tar.gz")
        url = "http://www.quantum-machine.org/datasets/iso17.tar.gz"

        try:
            request.urlretrieve(url, tarpath)
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, url)
            return False

        tar = tarfile.open(tarpath)
        tar.extractall(self.path)
        tar.close()

        # update metadata
        for fold in ISO17.existing_folds:
            dbpath = os.path.join(self.path, "iso17", fold + ".db")
            tmp_dbpath = os.path.join(tmpdir, "tmp.db")
            with connect(dbpath) as conn:
                with connect(tmp_dbpath) as tmp_conn:
                    tmp_conn.metadata = {
                        "_property_unit_dict": {
                            ISO17.energy: "eV",
                            ISO17.forces: "eV/Ang",
                        },
                        "_distance_unit": "Ang",
                        "atomrefs": {},
                    }
                    # add energy to data dict in db
                    for idx in tqdm(
                        range(len(conn)), f"parsing database file {dbpath}"
                    ):
                        atmsrw = conn.get(idx + 1)
                        data = atmsrw.data
                        data[ISO17.forces] = np.array(data[ISO17.forces])
                        data[ISO17.energy] = np.array([atmsrw.total_energy])
                        tmp_conn.write(atmsrw.toatoms(), data=data)

            os.remove(dbpath)
            os.rename(tmp_dbpath, dbpath)
        shutil.rmtree(tmpdir)

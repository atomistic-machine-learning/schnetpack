import logging
import os
import tarfile

import numpy as np
import torch
from ase.io import read

from schnetpack.data import (
    AtomsDataFormat,
    AtomsDataModule,
    AtomsDataModuleError,
    create_dataset,
    load_dataset,
)

__all__ = ["OrganicMaterialsDatabase"]


class OrganicMaterialsDatabase(AtomsDataModule):
    """
    Organic Materials Database (OMDB) of bulk organic crystals.
    Registration to the OMDB is free for academic users. This database contains DFT
    (PBE) band gap (OMDB-GAP1 database) for 12500 non-magnetic materials.

    References:

        .. [#omdb] Bart Olsthoorn, R. Matthias Geilhufe, Stanislav S. Borysov, Alexander V. Balatsky.
           Band gap prediction for large organic crystal structures with machine learning.
           https://arxiv.org/abs/1810.12814
    """

    BandGap = "band_gap"

    def __init__(
        self,
        datapath: str,
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
        raw_path: str | None = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to dataset
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
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            raw_path: path to raw tar.gz file with the data
        """
        super().__init__(
            datapath=datapath,
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
        self.raw_path = raw_path

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {OrganicMaterialsDatabase.BandGap: "eV"}

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )

            self._convert(dataset)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _convert(self, dataset):
        """
        Converts .tar.gz to a .db file
        """
        if self.raw_path is None or not os.path.exists(self.raw_path):
            # TODO: can we download here automatically like QM9?
            raise AtomsDataModuleError(
                "The path to the raw dataset is not provided or invalid and the db-file does "
                "not exist!"
            )
        logging.info("Converting %s to a .db file..", self.raw_path)
        tar = tarfile.open(self.raw_path, "r:gz")
        names = tar.getnames()
        tar.extractall()
        tar.close()

        structures = read("structures.xyz", index=":")
        Y = np.loadtxt("bandgaps.csv")
        [os.remove(name) for name in names]

        atoms_list = []
        property_list = []
        for i, at in enumerate(structures):
            atoms_list.append(at)
            property_list.append({OrganicMaterialsDatabase.BandGap: np.array([Y[i]])})
        dataset.add_systems(atoms_list=atoms_list, property_list=property_list)

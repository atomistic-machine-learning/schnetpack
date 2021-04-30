import os
from copy import copy
from typing import Optional, List, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from schnetpack.data import (
    AtomsDataFormat,
    resolve_format,
    load_dataset,
    BaseAtomsData,
    AtomsLoader,
    calculate_stats,
)

__all__ = ["AtomsDataModule", "AtomsDataModuleError"]


class AtomsDataModuleError(Exception):
    pass


class AtomsDataModule(pl.LightningDataModule):
    """
    Base class for atoms datamodules.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: int = None,
        num_val: int = None,
        num_test: int = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
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
            transforms: Preprocessing transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
        """
        super().__init__(
            train_transforms=train_transforms or copy(transforms) or [],
            val_transforms=val_transforms or copy(transforms) or [],
            test_transforms=test_transforms or copy(transforms) or [],
        )
        self._check_transforms(self.train_transforms)
        self._check_transforms(self.val_transforms)
        self._check_transforms(self.test_transforms)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size
        self.test_batch_size = test_batch_size or val_batch_size or batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.split_file = split_file
        self.datapath, self.format = resolve_format(datapath, format)
        self.load_properties = load_properties
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers or self.num_workers
        self.num_test_workers = num_test_workers or self.num_workers
        self.property_units = property_units
        self.distance_unit = distance_unit
        self._stats = {}

    def _check_transforms(self, transforms):
        for t in transforms:
            if not t.is_preprocessor:
                raise AtomsDataModuleError(
                    f"Transform of type {t} is not a preprocessor (is_preprocessor=False)!"
                )

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset(
            self.datapath,
            self.format,
            property_units=self.property_units,
            distance_unit=self.distance_unit,
        )

        # split dataset
        # TODO: handle IterDatasets

        if self.split_file is not None and os.path.exists(self.split_file):
            S = np.load(self.split_file)
            train_idx = S["train_idx"].tolist()
            val_idx = S["val_idx"].tolist()
            test_idx = S["test_idx"].tolist()
            if self.num_train and self.num_train != len(train_idx):
                raise AtomsDataModuleError(
                    f"Split file was given, but `num_train ({self.num_train}) != len(train_idx)` ({len(train_idx)})!"
                )
            if self.num_val and self.num_val != len(val_idx):
                raise AtomsDataModuleError(
                    f"Split file was given, but `num_val ({self.num_val}) != len(val_idx)` ({len(val_idx)})!"
                )
            if self.num_test and self.num_test != len(test_idx):
                raise AtomsDataModuleError(
                    f"Split file was given, but `num_test ({self.num_test}) != len(test_idx)` ({len(test_idx)})!"
                )
        else:
            if not self.num_train or not self.num_val:
                raise AtomsDataModuleError(
                    "If no `split_file` is given, "
                    + "the sizes of the training and validation partitions need to be set!"
                )

            if self.num_test is None:
                self.num_test = len(self.dataset) - self.num_train - self.num_val
            lengths = [self.num_train, self.num_val, self.num_test]
            offsets = torch.cumsum(torch.tensor(lengths), dim=0)
            indices = torch.randperm(sum(lengths)).tolist()
            train_idx, val_idx, test_idx = [
                indices[offset - length : offset]
                for offset, length in zip(offsets, lengths)
            ]
            np.savez(
                self.split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )

        self._train_dataset = self.dataset.subset(train_idx)
        self._val_dataset = self.dataset.subset(val_idx)
        self._test_dataset = self.dataset.subset(test_idx)

        # setup transforms
        for t in self.train_transforms:
            t.datamodule = self
        for t in self.val_transforms:
            t.datamodule = self
        for t in self.test_transforms:
            t.datamodule = self
        self._train_dataset.transforms = self.train_transforms
        self._val_dataset.transforms = self.val_transforms
        self._test_dataset.transforms = self.test_transforms

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats:
            return self._stats[key]

        stats = calculate_stats(
            self.train_dataloader(),
            divide_by_atoms={property: divide_by_atoms},
            atomref=self.train_dataset.atomrefs,
        )[property]
        self._stats[key] = stats
        return stats

    @property
    def train_dataset(self) -> BaseAtomsData:
        if not self.has_prepared_data:
            self.prepare_data()

        if not self.has_setup_fit:
            self.setup(stage="fit")
        return self._train_dataset

    @property
    def val_dataset(self) -> BaseAtomsData:
        if not self.has_prepared_data:
            self.prepare_data()

        if not self.has_setup_fit:
            self.setup(stage="fit")
        return self._val_dataset

    @property
    def test_dataset(self) -> BaseAtomsData:
        if not self.has_prepared_data:
            self.prepare_data()

        if not self.has_setup_fit:
            self.setup(stage="test")
        return self._test_dataset

    def train_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_test_workers,
            pin_memory=True,
        )

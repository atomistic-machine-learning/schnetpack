from typing import Optional, List

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split

import copy

from schnetpack.data import (
    AtomsDataFormat,
    resolve_format,
    load_dataset,
    BaseAtomsData,
    AtomsLoader,
)


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
        num_train: int,
        num_val: int,
        num_test: int = -1,
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transform_fn: Optional[torch.nn.Module] = None,
        train_transform_fn: Optional[torch.nn.Module] = None,
        val_transform_fn: Optional[torch.nn.Module] = None,
        test_transform_fn: Optional[torch.nn.Module] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
    ):
        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transform_fn: Transform applied to each system separately before batching.
            train_transform_fn: Overrides transform_fn for training.
            val_transform_fn: Overrides transform_fn for validation.
            test_transform_fn: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
        """
        super().__init__(
            train_transforms=train_transform_fn or transform_fn,
            val_transforms=val_transform_fn or transform_fn,
            test_transforms=test_transform_fn or transform_fn,
        )
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size
        self.test_batch_size = test_batch_size or val_batch_size or batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.datapath, self.format = resolve_format(datapath, format)
        self.load_properties = load_properties
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers or self.num_workers
        self.num_test_workers = num_test_workers or self.num_workers

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset(self.datapath, self.format)
        if self.num_test < 0:
            self.num_test = len(self.dataset) - self.num_train - self.num_val

        # TODO: handle IterDatasets

        lengths = [self.num_train, self.num_val, self.num_test]
        indices = torch.randperm(sum(lengths)).tolist()
        offsets = torch.cumsum(torch.tensor(lengths), dim=0)

        self._train_dataset, self._val_dataset, self._test_dataset = [
            self.dataset.subset(indices[offset - length : offset])
            for offset, length in zip(offsets, lengths)
        ]

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

    def train_dataloader(self):
        return AtomsLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return AtomsLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return AtomsLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_test_workers,
            pin_memory=True,
        )

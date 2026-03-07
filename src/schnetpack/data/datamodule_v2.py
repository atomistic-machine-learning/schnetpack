from __future__ import annotations

from copy import copy
from typing import List, Optional, Union, Dict, Any, Type

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import BatchSampler

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.provider import StatsAtomrefProvider
from schnetpack.data.splitting import RandomSplit, SplittingStrategy
from schnetpack.data.loader import AtomsLoader


class AtomsDataModuleV2(pl.LightningDataModule):
    """
    V2 DataModule:
      - accepts a dataset instance
      - handles splitting
      - builds StatsAtomrefProvider from train split
      - initializes transforms
    """

    def __init__(
        self,
        dataset: ASEAtomsData,
        batch_size: int,
        num_train: Union[int, float],
        num_val: Union[int, float],
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        splitting: Optional[SplittingStrategy] = None,
        transforms: Optional[List] = None,
        train_transforms: Optional[List] = None,
        val_transforms: Optional[List] = None,
        test_transforms: Optional[List] = None,
        num_workers: int = 0,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        train_sampler_cls: Optional[Type] = None,
        train_sampler_args: Optional[Dict[str, Any]] = None,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size
        self.test_batch_size = test_batch_size or val_batch_size or batch_size

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.split_file = split_file
        self.splitting = splitting or RandomSplit()
        self.num_workers = num_workers
        self._pin_memory = pin_memory

        self.train_transforms = train_transforms or copy(transforms) or []
        self.val_transforms = val_transforms or copy(transforms) or []
        self.test_transforms = test_transforms or copy(transforms) or []

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        self.provider: Optional[StatsAtomrefProvider] = None

        self.train_sampler_cls = train_sampler_cls
        self.train_sampler_args = train_sampler_args or {}

    @property
    def train_dataset(self) -> ASEAtomsData:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> ASEAtomsData:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_dataset

    @property
    def test_dataset(self) -> ASEAtomsData:
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_idx is None:
            self._load_partitions()

        self._train_dataset = self.dataset.subset(self.train_idx)
        self._val_dataset = self.dataset.subset(self.val_idx)
        self._test_dataset = self.dataset.subset(self.test_idx)

        self.provider = StatsAtomrefProvider(self._train_dataset)

        self._initialize_transform_list(self.train_transforms)
        self._initialize_transform_list(self.val_transforms)
        self._initialize_transform_list(self.test_transforms)

        self._train_dataset.transforms = self.train_transforms
        self._val_dataset.transforms = self.val_transforms
        self._test_dataset.transforms = self.test_transforms

    def _initialize_transform_list(self, transforms: List) -> None:
        if not transforms:
            return

        for t in transforms:
            t.initialize(provider=self.provider, atomrefs=self.provider.train_atomrefs)

    def _load_partitions(self) -> None:
        import os

        total_size = len(self.dataset)

        def _to_abs(x: Optional[Union[int, float]]) -> Optional[int]:
            if x is None:
                return None
            if isinstance(x, float) and x <= 1.0:
                return int(x * total_size)
            return int(x)

        num_train = _to_abs(self.num_train)
        num_val = _to_abs(self.num_val)
        num_test = _to_abs(self.num_test)

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        if self.split_file is not None and os.path.exists(self.split_file):
            split_data = np.load(self.split_file)
            self.train_idx = split_data["train_idx"].tolist()
            self.val_idx = split_data["val_idx"].tolist()
            self.test_idx = split_data["test_idx"].tolist()
            return

        if num_train is None or num_val is None:
            raise ValueError(
                "If no split file is given, num_train and num_val must be set."
            )

        train_idx, val_idx, test_idx = self.splitting.split(
            self.dataset, num_train, num_val, num_test
        )

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        if self.split_file is not None:
            np.savez(
                self.split_file,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )

    def _setup_sampler(self, sampler_cls, sampler_args, dataset):
        if sampler_cls is None:
            return None

        return BatchSampler(
            sampler=sampler_cls(
                data_source=dataset,
                num_samples=len(dataset),
                **sampler_args,
            ),
            batch_size=self.batch_size,
            drop_last=True,
        )

    def train_dataloader(self):
        if self._train_dataloader is None:
            train_batch_sampler = self._setup_sampler(
                sampler_cls=self.train_sampler_cls,
                sampler_args=self.train_sampler_args,
                dataset=self.train_dataset,
            )

            self._train_dataloader = AtomsLoader(
                self.train_dataset,
                batch_size=self.batch_size if train_batch_sampler is None else 1,
                shuffle=True if train_batch_sampler is None else False,
                batch_sampler=train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
            )

        return self._train_dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = AtomsLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
            )

        return self._val_dataloader

    def test_dataloader(self):
        if self._test_dataloader is None:
            self._test_dataloader = AtomsLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
            )

        return self._test_dataloader
from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl

from schnetpack.data.atoms import BaseAtomsData
from schnetpack.data.loader import AtomsLoader
from schnetpack.data.provider import StatsAtomrefProvider
from schnetpack.data.splitting import RandomSplit, SplittingStrategy


class AtomsDataModuleV2(pl.LightningDataModule):
    """
    V2 DataModule:
      - accepts a dataset instance (datasets are independent)
      - handles splitting + loaders/batching
      - builds StatsAtomrefProvider from train split
      - initializes transforms via t.initialize(provider, atomrefs=...)
    """

    def __init__(
        self,
        dataset: BaseAtomsData,
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
        strict_transform_init: bool = True,
        loader_kwargs: Optional[Dict[str, Any]] = None,
        val_loader_kwargs: Optional[Dict[str, Any]] = None,
        test_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # swallow legacy knobs without breaking configs
    ):
        super().__init__()

        if kwargs:
            pass

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        self.split_file = split_file
        self.splitting = splitting or RandomSplit()

        self._train_transforms = train_transforms or copy(transforms) or []
        self._val_transforms = val_transforms or copy(transforms) or []
        self._test_transforms = test_transforms or copy(transforms) or []
        self.strict_transform_init = strict_transform_init

        self.num_workers = num_workers

        self.loader_kwargs = loader_kwargs or {}
        self.val_loader_kwargs = val_loader_kwargs or {}
        self.test_loader_kwargs = test_loader_kwargs or {}

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        self.provider: Optional[StatsAtomrefProvider] = None

    @property
    def train_transforms(self):
        return self._train_transforms

    @property
    def val_transforms(self):
        return self._val_transforms

    @property
    def test_transforms(self):
        return self._test_transforms

    @property
    def train_dataset(self) -> BaseAtomsData:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> BaseAtomsData:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_dataset

    @property
    def test_dataset(self) -> BaseAtomsData:
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_idx is None:
            self._load_partitions()

        self._train_dataset = self.dataset.subset(self.train_idx)
        self._val_dataset = self.dataset.subset(self.val_idx)
        self._test_dataset = self.dataset.subset(self.test_idx)

        train_atomrefs = getattr(self._train_dataset, "atomrefs", None)

        self.provider = StatsAtomrefProvider(
            train_dataloader_factory=self.train_dataloader,
            train_atomrefs=train_atomrefs,
        )

        self._initialize_transform_list(
            self.train_transforms, train_atomrefs=train_atomrefs
        )
        self._initialize_transform_list(
            self.val_transforms, train_atomrefs=train_atomrefs
        )
        self._initialize_transform_list(
            self.test_transforms, train_atomrefs=train_atomrefs
        )

        self._train_dataset.transforms = self.train_transforms
        self._val_dataset.transforms = self.val_transforms
        self._test_dataset.transforms = self.test_transforms

    def _initialize_transform_list(self, transforms: List, train_atomrefs):
        if not transforms:
            return
        for t in transforms:
            init_fn = getattr(t, "initialize", None)
            if callable(init_fn):
                init_fn(self.provider, atomrefs=train_atomrefs)
                continue
            if self.strict_transform_init:
                raise RuntimeError(
                    f"Transform {type(t).__name__} does not implement initialize(provider, atomrefs=...)."
                )

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

        self.num_train, self.num_val, self.num_test = num_train, num_val, num_test

        if self.split_file is not None and os.path.exists(self.split_file):
            S = np.load(self.split_file)
            self.train_idx = S["train_idx"].tolist()
            self.val_idx = S["val_idx"].tolist()
            self.test_idx = S["test_idx"].tolist()
            return

        train_idx, val_idx, test_idx = self.splitting.split(
            self.dataset, num_train, num_val, num_test
        )
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

        if self.split_file is not None:
            np.savez(
                self.split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )

    def train_dataloader(self) -> AtomsLoader:
        if self._train_loader is None:
            self._train_loader = AtomsLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                **self.loader_kwargs,
            )
        return self._train_loader

    def val_dataloader(self) -> AtomsLoader:
        if self._val_loader is None:
            self._val_loader = AtomsLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                **{**self.loader_kwargs, **self.val_loader_kwargs},
            )
        return self._val_loader

    def test_dataloader(self) -> AtomsLoader:
        if self._test_loader is None:
            self._test_loader = AtomsLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                **{**self.loader_kwargs, **self.test_loader_kwargs},
            )
        return self._test_loader

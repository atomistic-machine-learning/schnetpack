import logging
import numpy as np
import os
import pytorch_lightning as pl
import schnetpack as spk
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import (
    random_split,
    Subset,
)
from torch.utils.data.sampler import (
    RandomSampler,
)
from typing import (
    List,
)


class AtomsDataModule(pl.LightningDataModule):
    """ Lightning data module for SchNetPack atoms data """

    def __init__(
        self,
        dataset: spk.data.AtomsData,
        num_train: int,
        num_val: int,
        batch_size: int,
        val_batch_size: int = None,
        test_batch_size: int = None,
        copy2tmp: bool = False,
        split_file: str = None,
        properties: List[str] = None,
        name: str = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_train = num_train
        self.num_val = num_val
        self.properties = properties

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size

        self.copy2tmp = copy2tmp
        self.split_file = split_file
        self.tmpdirs = []
        self.name = name

    def setup(
        self,
        stage=None,
    ):
        if self.split_file is not None and os.path.exists(self.split_file):
            logging.info("Load split.")
            S = np.load(self.split_file)
            train_idx = S["train_idx"].tolist()
            val_idx = S["val_idx"].tolist()
            test_idx = S["test_idx"].tolist()
            self.train_data = Subset(
                self.dataset,
                indices=train_idx,
            )
            self.val_data = Subset(
                self.dataset,
                indices=val_idx,
            )
            self.test_data = Subset(
                self.dataset,
                indices=test_idx,
            )
        else:
            (self.train_data, self.val_data, self.test_data,) = random_split(
                self.dataset,
                [
                    self.num_train,
                    self.num_val,
                    len(self.dataset) - self.num_train - self.num_val,
                ],
            )
            if self.split_file is not None:
                np.savez(
                    self.split_file,
                    train_idx=self.train_data.indices,
                    val_idx=self.val_data.indices,
                    test_idx=self.test_data.indices,
                )

    def train_dataloader(
        self,
    ):
        return spk.data.AtomsLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=RandomSampler(self.train_data),
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(
        self,
    ):
        return spk.data.AtomsLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(
        self,
    ):
        return spk.data.AtomsLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            num_workers=2,
            pin_memory=True,
        )

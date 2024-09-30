import logging
import os
import shutil
from copy import copy
from typing import Optional, List, Dict, Tuple, Union, Any, Type
import numpy as np
import fasteners
import pytorch_lightning as pl
import torch
from torch.utils.data import BatchSampler

from schnetpack.data import (
    AtomsDataFormat,
    resolve_format,
    load_dataset,
    BaseAtomsData,
    AtomsLoader,
    calculate_stats,
    SplittingStrategy,
    RandomSplit,
)


__all__ = ["AtomsDataModule", "AtomsDataModuleError"]


class AtomsDataModuleError(Exception):
    pass


class AtomsDataModule(pl.LightningDataModule):
    """
    A general ``LightningDataModule`` for SchNetPack datasets.

    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        train_sampler_cls: Optional[Type] = None,
        train_sampler_args: Optional[Dict[str, Any]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = False,
    ):
        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples (absolute or relative)
            num_val: number of validation examples (absolute or relative)
            num_test: number of test examples (absolute or relative)
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then
                batch_size.
            transforms: Preprocessing transform applied to each system separately before
                batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            train_sampler_cls: type of torch training sampler.
                This is by default wrapped into a torch.utils.data.BatchSampler.
            train_sampler_args: dict of train_sampler keyword arguments.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers
                (overrides num_workers).
            num_test_workers: Number of test data loader workers
                (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string
                (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string
                (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. to a local file
                system for faster performance.
            cleanup_workdir_stage: Determines after which stage to remove the data
                workdir
            splitting: Method to generate train/validation/test partitions
                (default: RandomSplit)
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                set to true, when GPUs are used.
        """
        super().__init__()
        self._train_transforms = train_transforms or copy(transforms) or []
        self._val_transforms = val_transforms or copy(transforms) or []
        self._test_transforms = test_transforms or copy(transforms) or []

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size
        self.test_batch_size = test_batch_size or val_batch_size or batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.splitting = splitting or RandomSplit()
        self.split_file = split_file
        self.datapath, self.format = resolve_format(datapath, format)
        self.load_properties = load_properties
        self.num_workers = num_workers
        self.num_val_workers = self.num_workers
        self.num_test_workers = self.num_workers
        if num_val_workers is not None:
            self.num_val_workers = num_val_workers
        if num_test_workers is not None:
            self.num_test_workers = num_test_workers
        self.property_units = property_units
        self.distance_unit = distance_unit
        self._stats = {}
        self._is_setup = False
        self.data_workdir = data_workdir
        self.cleanup_workdir_stage = cleanup_workdir_stage
        self._pin_memory = pin_memory

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.dataset = None
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        self.train_sampler_cls = train_sampler_cls
        self.train_sampler_args = train_sampler_args

    @property
    def train_transforms(self):
        """
        Optional transforms (or collection of transforms) you can apply to train
        dataset.
        """
        return self._train_transforms

    @property
    def val_transforms(self):
        """
        Optional transforms (or collection of transforms) you can apply to validation
        dataset.
        """
        return self._val_transforms

    @property
    def test_transforms(self):
        """
        Optional transforms (or collection of transforms) you can apply to test dataset.
        """
        return self._test_transforms

    def setup(self, stage: Optional[str] = None):
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
            )

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            self._test_dataset = self.dataset.subset(self.test_idx)
            self._setup_transforms()

    def _copy_to_workdir(self):
        """
        Copies the data to given (fast) working location. Useful for working on cluster
        with slow shared and fast local file systems.

        Returns:
            path to data in workdir
        """
        if not os.path.exists(self.data_workdir):
            os.makedirs(self.data_workdir, exist_ok=True)
        name = self.datapath.split("/")[-1]
        datapath = os.path.join(self.data_workdir, name)
        lock = fasteners.InterProcessLock(
            os.path.join(self.data_workdir, f"dataworkdir_{name}.lock")
        )
        with lock:
            self._log_with_rank("Enter lock")

            # retry reading, in case other process finished in the meantime
            if not os.path.exists(datapath):
                self._log_with_rank("Copy data to data workdir")
                shutil.copy(self.datapath, datapath)

            # reset datasets in case they need to be reloaded
            self.dataset = None
            self._train_dataset = None
            self._val_dataset = None
            self._test_dataset = None

            # reset cleanup
            self._has_teardown_fit = False
            self._has_teardown_val = False
            self._has_teardown_test = False
        self._log_with_rank("Exited lock")
        return datapath

    def teardown(self, stage: Optional[str] = None):
        if self.cleanup_workdir_stage and stage == self.cleanup_workdir_stage:
            if self.data_workdir is not None:
                try:
                    shutil.rmtree(self.data_workdir)
                except:
                    pass
                self._has_setup_fit = False
                self._has_setup_val = False
                self._has_setup_test = False

        # teardown transforms
        for t in self.train_transforms:
            t.teardown()
        for t in self.val_transforms:
            t.teardown()
        for t in self.test_transforms:
            t.teardown()

    def _load_partitions(self):
        # split dataset
        lock = fasteners.InterProcessLock("splitting.lock")

        with lock:
            self._log_with_rank("Enter splitting lock")

            if self.split_file is not None and os.path.exists(self.split_file):
                self._log_with_rank("Load split")

                S = np.load(self.split_file)
                self.train_idx = S["train_idx"].tolist()
                self.val_idx = S["val_idx"].tolist()
                self.test_idx = S["test_idx"].tolist()
                if self.num_train and self.num_train != len(self.train_idx):
                    logging.warning(
                        f"Split file was given, but `num_train ({self.num_train})"
                        + f" != len(train_idx)` ({len(self.train_idx)})!"
                    )
                if self.num_val and self.num_val != len(self.val_idx):
                    logging.warning(
                        f"Split file was given, but `num_val ({self.num_val})"
                        + f" != len(val_idx)` ({len(self.val_idx)})!"
                    )
                if self.num_test and self.num_test != len(self.test_idx):
                    logging.warning(
                        f"Split file was given, but `num_test ({self.num_test})"
                        + f" != len(test_idx)` ({len(self.test_idx)})!"
                    )
            else:
                self._log_with_rank("Create split")

                if not self.num_train or not self.num_val:
                    raise AtomsDataModuleError(
                        "If no `split_file` is given, the sizes of the training and"
                        + " validation partitions need to be set!"
                    )

                self.train_idx, self.val_idx, self.test_idx = self.splitting.split(
                    self.dataset, self.num_train, self.num_val, self.num_test
                )

                if self.split_file is not None:
                    self._log_with_rank("Save split")
                    np.savez(
                        self.split_file,
                        train_idx=self.train_idx,
                        val_idx=self.val_idx,
                        test_idx=self.test_idx,
                    )

        self._log_with_rank("Exit splitting lock")

    def _log_with_rank(self, msg: str):
        if self.trainer is not None:
            logging.debug(
                "Global rank:",
                self.trainer.global_rank,
                ", lokal rank:",
                self.trainer.local_rank,
                " >> ",
                msg,
            )
        else:
            logging.debug(">> ", msg)

    def _setup_sampler(self, sampler_cls, sampler_args, dataset):
        if sampler_cls is None:
            return None
        else:
            batch_sampler = BatchSampler(
                sampler=sampler_cls(
                    data_source=dataset,
                    num_samples=len(dataset),
                    **sampler_args,
                ),
                batch_size=self.batch_size,
                drop_last=True,
            )
            return batch_sampler

    def _setup_transforms(self):
        for t in self.train_transforms:
            t.datamodule(self)
        for t in self.val_transforms:
            t.datamodule(self)
        for t in self.test_transforms:
            t.datamodule(self)
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
            atomref=self.train_dataset.atomrefs if remove_atomref else None,
        )[property]
        self._stats[key] = stats
        return stats

    @property
    def train_dataset(self) -> BaseAtomsData:
        return self._train_dataset

    @property
    def val_dataset(self) -> BaseAtomsData:
        return self._val_dataset

    @property
    def test_dataset(self) -> BaseAtomsData:
        return self._test_dataset

    def train_dataloader(self) -> AtomsLoader:
        if self._train_dataloader is None:

            train_batch_sampler = self._setup_sampler(
                sampler_cls=self.train_sampler_cls,
                sampler_args=self.train_sampler_args,
                dataset=self._train_dataset,
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

    def val_dataloader(self) -> AtomsLoader:
        if self._val_dataloader is None:
            self._val_dataloader = AtomsLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_val_workers,
                pin_memory=self._pin_memory,
            )
        return self._val_dataloader

    def test_dataloader(self) -> AtomsLoader:
        if self._test_dataloader is None:
            self._test_dataloader = AtomsLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                pin_memory=self._pin_memory,
            )
        return self._test_dataloader

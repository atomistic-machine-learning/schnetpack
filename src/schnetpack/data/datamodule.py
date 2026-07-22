from typing import Optional, Union, Dict, Any, Type, Tuple
import os

import fasteners
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import BatchSampler

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.provider import StatsAtomrefProvider, train_partition_fingerprint
from schnetpack.data.splitting import SPLITTING_LOCK, RandomSplit, SplittingStrategy
from schnetpack.data.loader import AtomsLoader

__all__ = ["AtomsDataModule"]

# Default for stats_file: derive the path from split_file. A sentinel (not
# None) because None means "persistence off".
_DERIVE_STATS_FILE = "<derive from split_file>"


class AtomsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule for SchNetPack datasets. It

      - accepts a prebuilt :class:`ASEAtomsData` instance,
      - handles train/val/test splitting,
      - builds a stats provider from the train split,
      - initializes the dataset transforms with it.
    """

    # Arguments of the removed legacy AtomsDataModule that are now configured
    # on the dataset (or were dropped). Rejected loudly, because silently
    # ignoring e.g. `property_units` or `transforms` would produce wrong
    # results without any error.
    _LEGACY_ARGS = {
        "datapath",
        "format",
        "load_properties",
        "transforms",
        "train_transforms",
        "val_transforms",
        "test_transforms",
        "num_val_workers",
        "num_test_workers",
        "property_units",
        "distance_unit",
        "data_workdir",
        "cleanup_workdir_stage",
    }

    def __init__(
        self,
        dataset: ASEAtomsData,
        batch_size: int,
        num_train: Union[int, float],
        num_val: Union[int, float],
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        stats_file: Optional[str] = _DERIVE_STATS_FILE,
        splitting: Optional[SplittingStrategy] = None,
        num_workers: int = 0,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        train_sampler_cls: Optional[Type] = None,
        train_sampler_args: Optional[Dict[str, Any]] = None,
        pin_memory: bool = False,
        provider: Optional[Type] = None,
        **kwargs,
    ):
        """
        Args:
            dataset: prebuilt ASEAtomsData dataset instance
            batch_size: (train) batch size
            num_train: number of training examples (absolute or relative)
            num_val: number of validation examples (absolute or relative)
            num_test: number of test examples (absolute or relative)
            split_file: path to npz file with data partitions
            stats_file: path to the npz file persisting training statistics
                and estimated atomrefs, keyed by the train-partition
                fingerprint. By default derived from split_file
                (<split>_stats.npz next to it). Set to None to disable
                persistence; reruns then recompute statistics.
            splitting: Method to generate train/validation/test partitions
                    (default: RandomSplit)
            num_workers: Number of data loader workers
            val_batch_size: validation batch size. If None, use test_batch_size, then
                batch_size
            test_batch_size: test batch size. If None, use val_batch_size, then
                batch_size
            train_sampler_cls: type of torch training sampler.
                This is by default wrapped into a torch.utils.data.BatchSampler.
            train_sampler_args: dict of train_sampler keyword arguments.
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                    set to true, when GPUs are used.
            provider: stats provider class, constructed during setup() from
                the base dataset and the train index list. If None, use
                StatsAtomrefProvider.
        """
        # Unknown kwargs other than the legacy arguments are tolerated
        # silently, because hydra data configs use top-level keys as
        # interpolation variables (e.g. `molecule`, `fold`).
        used_legacy_args = sorted(self._LEGACY_ARGS & kwargs.keys())
        if used_legacy_args:
            raise TypeError(
                f"The following arguments are no longer supported by "
                f"`AtomsDataModule`: {used_legacy_args}. Configure them on "
                "the dataset instance instead (e.g. in the `dataset:` block of "
                "the data config)."
            )
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size
        self.test_batch_size = test_batch_size or val_batch_size or batch_size

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.split_file = split_file
        if stats_file == _DERIVE_STATS_FILE:
            stats_file = (
                os.path.splitext(split_file)[0] + "_stats.npz"
                if split_file is not None
                else None
            )
        self.stats_file = stats_file
        self.splitting = splitting or RandomSplit()
        self.num_workers = num_workers
        self._pin_memory = pin_memory

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.train_fingerprint: Optional[str] = None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        self._provider_cls = provider or StatsAtomrefProvider
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
        # Lightning calls setup() once per stage; recreating the subsets and
        # the stats provider would drop the statistics cache between stages.
        if self._train_dataset is not None:
            return

        if self.train_idx is None:
            self._load_partitions()

        # Statistics are a pure function of (dataset, train partition); the
        # fingerprint identifies that partition, e.g. for persisted stats.
        self.train_fingerprint = train_partition_fingerprint(
            len(self.dataset), self.train_idx
        )

        # The split label activates the per-split transform selection of
        # ASEAtomsData for anyone touching the subsets directly.
        self._train_dataset = self.dataset.subset(self.train_idx, split="train")
        self._val_dataset = self.dataset.subset(self.val_idx, split="val")
        self._test_dataset = self.dataset.subset(self.test_idx, split="test")

        self.provider = self._provider_cls(
            self.dataset,
            self.train_idx,
            stats_file=self.stats_file,
            fingerprint=self.train_fingerprint,
        )

        self._train_dataset.initialize_transforms(self.provider)
        self._val_dataset.initialize_transforms(self.provider)
        self._test_dataset.initialize_transforms(self.provider)

    def teardown(self, stage: Optional[str] = None) -> None:
        # Transforms with external resources (e.g. cached neighbor lists)
        # rely on teardown() being called.
        for ds in (self._train_dataset, self._val_dataset, self._test_dataset):
            if ds is None:
                continue
            for t in ds.get_split_transforms():
                t.teardown()

    # The datamodule itself satisfies the stats-source protocol: model
    # postprocessors (e.g. AddOffsets) are initialized late via
    # `initialize(datamodule)` and must read from the *same* cached provider
    # as the data-pipeline transforms — otherwise they would recompute
    # statistics on already-transformed data and end up with wrong offsets.
    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.provider is None:
            raise RuntimeError("Call setup() before accessing statistics.")
        return self.provider.get_stats(property, divide_by_atoms, remove_atomref)

    def get_atomrefs(
        self, property: str, is_extensive: bool, estimate: bool = True
    ) -> Dict[str, torch.Tensor]:
        if self.provider is None:
            raise RuntimeError("Call setup() before accessing atomrefs.")
        return self.provider.get_atomrefs(property, is_extensive, estimate)

    def _load_partitions(self) -> None:
        # Serialize split creation with an inter-process lock, so concurrent
        # DDP ranks / jobs cannot race on writing split.npz and end up with
        # different partitions per rank.
        lock = fasteners.InterProcessLock(SPLITTING_LOCK)

        with lock:
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

                # Validate a pre-existing split file against the requested
                # sizes; a stale split.npz would otherwise be used silently.
                if num_train and num_train != len(self.train_idx):
                    raise ValueError(
                        f"Split file was given, but `num_train ({num_train})"
                        f" != len(train_idx)` ({len(self.train_idx)})!"
                    )
                if num_val and num_val != len(self.val_idx):
                    raise ValueError(
                        f"Split file was given, but `num_val ({num_val})"
                        f" != len(val_idx)` ({len(self.val_idx)})!"
                    )
                if num_test and num_test != len(self.test_idx):
                    raise ValueError(
                        f"Split file was given, but `num_test ({num_test})"
                        f" != len(test_idx)` ({len(self.test_idx)})!"
                    )
                return

            if num_train is None or num_val is None:
                raise ValueError("num_train and num_val must be set if no split file.")

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

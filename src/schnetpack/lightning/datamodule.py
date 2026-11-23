import logging
import os
from contextlib import nullcontext
from typing import Any

import fasteners
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import BatchSampler

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.cache import ArrayStats, AtomsCache
from schnetpack.data.loader import AtomsLoader
from schnetpack.data.provider import StatsAtomrefProvider, train_partition_fingerprint
from schnetpack.data.sampler import BudgetBatches
from schnetpack.data.splitting import RandomSplit, SplittingStrategy, splitting_lock
from schnetpack.transform.base import Transform

__all__ = ["AtomsDataModule"]

log = logging.getLogger(__name__)

# Default for stats_file: derive the path from split_file. A sentinel (not
# None) because None means "persistence off".
_DERIVE_STATS_FILE = "<derive from split_file>"


class AtomsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule for SchNetPack datasets. It

      - accepts a prebuilt :class:`ASEAtomsData` instance,
      - handles train/val/test splitting,
      - builds a stats provider from the train split,
      - initializes the dataset transforms and the batch transforms with it,
      - builds an :class:`~schnetpack.data.AtomsLoader` for each split, which moves
        the batches to the trainer's device and applies the batch transforms.

    By default, DataLoader workers read the structures with the dataset's transforms
    on every read. With ``cache=True`` they are read once instead
    (:class:`~schnetpack.data.AtomsCache`, kept in ``cache_dir``) and the loaders
    slice whole padded batches from the cache: no workers and no reads per
    structure, so small models are not left waiting for data. The dataset's
    transforms must then act on one structure and be deterministic (centering, a
    neighbor list); offsets, casts and noise go into ``batch_transforms``, which
    serve both routes. Padded batches mark their padding rows by
    ``properties.atom_mask`` and ``properties.structure_mask``, which
    :class:`~schnetpack.lightning.AtomisticTask` leaves out of losses and metrics.
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
        num_train: int | float,
        num_val: int | float,
        num_test: int | float | None = None,
        split_file: str | None = "split.npz",
        stats_file: str | None = _DERIVE_STATS_FILE,
        splitting: SplittingStrategy | None = None,
        num_workers: int = 0,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_sampler_cls: type | None = None,
        train_sampler_args: dict[str, Any] | None = None,
        pin_memory: bool = False,
        provider: type | None = None,
        batch_transforms: list[Transform] | None = None,
        cache: bool = False,
        val_seed: int | None = 0,
        cache_dir: str | None = None,
        drop_last: bool = True,
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
                StatsAtomrefProvider. With a cache: ArrayStats on the cache.
            batch_transforms: transforms of every collated batch (offsets, casts,
                noise), on the trainer's device
            cache: read every structure once into an
                :class:`~schnetpack.data.AtomsCache` and slice padded batches from
                it (training packs them to atom and pair budgets). Takes no train
                sampler, and the dataset's transforms must not move atoms or need
                statistics
            val_seed: the seed of random batch transforms (noise) in validation
                and testing: the same draws every epoch. None: new draws
            cache_dir: with a cache, the folder it is kept in: loaded if it holds
                one, else made and saved there. None: made in memory every run
            drop_last: with a cache, training drops the last packed batch of each
                pass (usually not full), also a full one. False keeps it: every
                training structure once per pass
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
        self.train_fingerprint: str | None = None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        #: the loader of each split ("train", "val", "test"), built in setup()
        self.loaders: dict[str, AtomsLoader] = {}

        self._provider_cls = provider or StatsAtomrefProvider
        self.provider: StatsAtomrefProvider | None = None

        self.train_sampler_cls = train_sampler_cls
        self.train_sampler_args = train_sampler_args or {}

        if cache and train_sampler_cls is not None:
            raise ValueError("cache=True does not take a train sampler")
        self.batch_transforms = list(batch_transforms or [])
        self.cache = cache
        self.val_seed = val_seed
        self.cache_dir = cache_dir
        self.drop_last = drop_last
        #: with cache=True: the cache, made or loaded in setup()
        self.atoms_cache: AtomsCache | None = None

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

    def setup(self, stage: str | None = None) -> None:
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

        if self.cache:
            self.atoms_cache = self._make_cache()
            positions = self.atoms_cache.positions(self._train_dataset)
            self.provider = ArrayStats(
                self.atoms_cache, positions, self.dataset.atomrefs
            )
        else:
            self.provider = self._provider_cls(
                self.dataset,
                self.train_idx,
                stats_file=self.stats_file,
                fingerprint=self.train_fingerprint,
            )

        self._train_dataset.initialize_transforms(self.provider)
        self._val_dataset.initialize_transforms(self.provider)
        self._test_dataset.initialize_transforms(self.provider)
        for t in self.batch_transforms:
            t.initialize(self.provider)
        self._setup_loaders()

    def _make_cache(self) -> AtomsCache:
        """The dataset's structures, read once: loaded from cache_dir if it holds
        them, else made (and saved there)."""
        ds = self.dataset
        if any(
            getattr(ds, f"{split}_transforms", None) is not None
            for split in ("train", "val", "test")
        ):
            raise ValueError(
                "cache=True applies the dataset's transforms to all splits: "
                "per-split transform lists are not supported"
            )
        if self.cache_dir is None:
            return AtomsCache.from_dataset(ds, self.num_workers)
        # under a file lock: of several processes (DDP) one makes the cache
        cache_dir = os.fspath(self.cache_dir).rstrip("/")
        with fasteners.InterProcessLock(cache_dir + ".lock"):
            if os.path.exists(os.path.join(cache_dir, "meta.json")):
                cache = AtomsCache.load(cache_dir)
                if len(cache) != len(ds):
                    raise ValueError(
                        f"the cache in {cache_dir} holds {len(cache)} structures, "
                        f"the dataset {len(ds)}: delete the folder to make it again"
                    )
                return cache
            cache = AtomsCache.from_dataset(ds, self.num_workers)
            cache.save(cache_dir)
            return cache

    def _setup_loaders(self) -> None:
        """The loader of each split, after the transforms are initialized; the
        loaders move the batches to the trainer's device."""
        device = None if self.trainer is None else self.trainer.strategy.root_device
        cached = self.atoms_cache is not None
        common = dict(
            cache=self.atoms_cache,
            batch_transforms=self.batch_transforms,
            device=device,
        )
        sampler, train_args = None, {}
        if cached:
            train_args = dict(drop_last=self.drop_last)
        else:  # workers read the structures
            common.update(num_workers=self.num_workers, pin_memory=self._pin_memory)
            sampler = self._setup_sampler(
                sampler_cls=self.train_sampler_cls,
                sampler_args=self.train_sampler_args,
                dataset=self._train_dataset,
            )
        # the seed of the cached route's training order follows seed_everything
        seed = int(torch.randint(2**31, ())) if cached else 0
        self.loaders = {
            "train": AtomsLoader(
                self._train_dataset,
                batch_size=self.batch_size if sampler is None else 1,
                shuffle=sampler is None,
                batch_sampler=sampler,
                seed=seed,
                **train_args,
                **common,
            ),
            "val": AtomsLoader(
                self._val_dataset,
                batch_size=self.val_batch_size,
                transform_seed=self.val_seed,
                **common,
            ),
            "test": AtomsLoader(
                self._test_dataset,
                batch_size=self.test_batch_size,
                transform_seed=self.val_seed,
                **common,
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        # With cache=True, the training order comes from a generator of its own
        # (BudgetBatches), which a resumed run has to continue. Without a cache,
        # the order comes from torch's global generator, which the task restores.
        batches = getattr(self.loaders.get("train"), "batch_sampler", None)
        if isinstance(batches, BudgetBatches):
            return {"train_order": batches.rng.bit_generator.state}
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Lightning calls this after setup(), which made the loaders
        if "train_order" in state_dict:
            batches = self.loaders["train"].batch_sampler
            batches.rng.bit_generator.state = state_dict["train_order"]

    def teardown(self, stage: str | None = None) -> None:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.provider is None:
            raise RuntimeError("Call setup() before accessing statistics.")
        return self.provider.get_stats(property, divide_by_atoms, remove_atomref)

    def get_atomrefs(
        self, property: str, is_extensive: bool, estimate: bool = True
    ) -> dict[str, torch.Tensor]:
        if self.provider is None:
            raise RuntimeError("Call setup() before accessing atomrefs.")
        return self.provider.get_atomrefs(property, is_extensive, estimate)

    def _load_partitions(self) -> None:
        # Serialize split creation with an inter-process lock, so concurrent
        # DDP ranks / jobs cannot race on writing split.npz and end up with
        # different partitions per rank. Without a split file nothing is
        # shared between processes, so there is nothing to lock.
        lock = (
            splitting_lock(self.split_file)
            if self.split_file is not None
            else nullcontext()
        )

        with lock:
            total_size = len(self.dataset)

            def _to_abs(x: int | float | None) -> int | None:
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
                log.info(f"Loading existing split from {self.split_file}")
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
                log.info(f"Writing new split to {self.split_file}")
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
                **{"num_samples": len(dataset), **sampler_args},
            ),
            batch_size=self.batch_size,
            drop_last=False,
        )

    def train_dataloader(self) -> AtomsLoader:
        return self._loader("train")

    def val_dataloader(self) -> AtomsLoader:
        return self._loader("val")

    def test_dataloader(self) -> AtomsLoader:
        return self._loader("test")

    def _loader(self, split: str) -> AtomsLoader:
        if split not in self.loaders:
            raise RuntimeError(f"Call setup() before accessing the {split} loader.")
        return self.loaders[split]

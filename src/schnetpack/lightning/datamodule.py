import json
import logging
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import BatchSampler

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.device import (
    ARRAYS_FORMAT,
    BudgetBatches,
    DeviceAtomsData,
    OrderedBatches,
    ShuffledBatches,
    convert_to_arrays,
    dataset_fingerprint,
    neighbor_list_descriptor,
)
from schnetpack.data.loader import AtomsLoader
from schnetpack.data.provider import StatsAtomrefProvider, train_partition_fingerprint
from schnetpack.data.splitting import RandomSplit, SplittingStrategy, splitting_lock
from schnetpack.transform.neighborlist import (
    AllToAllNeighborList,
    NeighborListTransform,
)

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
      - initializes the dataset transforms with it.

    By default, DataLoader workers read and transform the structures. With
    ``data_device`` (e.g. "cuda") the dataset is held on that device instead and
    batches are built there (:mod:`schnetpack.data.device`): no workers and no reads
    per structure, so small models are not left waiting for data. The dataset is
    converted once into flat arrays (cached in ``array_dir``). Its neighbor list
    transform runs once per structure at that conversion; its other transforms run on
    whole batches on the device and must accept collated batches (as the centering,
    offset, cast and ``Diffuse`` transforms do). Batches have fixed shapes: their
    padding rows are marked by ``properties.atom_mask`` and
    ``properties.structure_mask``, which :class:`~schnetpack.lightning.AtomisticTask`
    leaves out of losses and metrics.
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
        data_device: str | None = None,
        batching: str = "fixed",
        pad_k: float = 6.0,
        val_seed: int | None = 0,
        array_dir: str | None = None,
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
            data_device: hold the dataset on this device (e.g. "cuda") and build
                the batches there; None: DataLoader workers
            batching: with data_device, "fixed" (batch_size structures per training
                batch, padded for the largest of them) or "budget" (batches packed
                to the atoms and pairs batch_size structures hold on average, with
                little padding)
            pad_k: with data_device and fixed batching, the training pad sizes in
                standard deviations above the mean; rarer batches are skipped
            val_seed: with data_device, the seed of random transforms (noise) in
                validation and testing: the same draws every epoch. None: new draws
            array_dir: with data_device, where the converted arrays are cached
                (default: ``<datapath>.arrays``)
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

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        self._provider_cls = provider or StatsAtomrefProvider
        self.provider: StatsAtomrefProvider | None = None

        self.train_sampler_cls = train_sampler_cls
        self.train_sampler_args = train_sampler_args or {}

        if batching not in ("fixed", "budget"):
            raise ValueError(f"batching must be 'fixed' or 'budget', not {batching!r}")
        if data_device is None and batching != "fixed":
            raise ValueError("batching='budget' needs data_device")
        if data_device is not None and train_sampler_cls is not None:
            raise ValueError("data_device does not take a train sampler")
        self.data_device = data_device
        self.batching = batching
        self.pad_k = pad_k
        self.val_seed = val_seed
        self.array_dir = array_dir
        self.data: DeviceAtomsData | None = None
        # set by a training step that builds its own batches from structure
        # indices (AtomisticTask with cuda_graph)
        self.train_batches_in_step = False

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

        if self.data_device is None:
            self.provider = self._provider_cls(
                self.dataset,
                self.train_idx,
                stats_file=self.stats_file,
                fingerprint=self.train_fingerprint,
            )
        else:
            self._setup_device()

        self._train_dataset.initialize_transforms(self.provider)
        self._val_dataset.initialize_transforms(self.provider)
        self._test_dataset.initialize_transforms(self.provider)
        if self.data_device is not None:
            # their buffers (e.g. RemoveOffsets' atomrefs) on the batches' device
            for ts in self._batch_transforms.values():
                for t in ts:
                    t.to(self.data.device)

    def _setup_device(self) -> None:
        """Convert the dataset (once; cached), hold it on data_device, and set up
        the training batches and the pad sizes."""
        neighbor_lists, self._batch_transforms = {}, {}
        for stage, ds in (
            ("train", self._train_dataset),
            ("val", self._val_dataset),
            ("test", self._test_dataset),
        ):
            neighbor_lists[stage], self._batch_transforms[stage] = _split_transforms(
                ds.get_split_transforms()
            )
        descriptors = {
            json.dumps(neighbor_list_descriptor(nl)) for nl in neighbor_lists.values()
        }
        if len(descriptors) > 1:
            raise ValueError("data_device needs the same neighbor list for all splits")
        neighbor_list = neighbor_lists["train"]

        array_dir = self.array_dir or f"{self.dataset.datapath}.arrays"
        wanted = {
            "source": json.loads(json.dumps(dataset_fingerprint(self.dataset))),
            "format": ARRAYS_FORMAT,
            "neighbor_list": neighbor_list_descriptor(neighbor_list),
        }
        meta_file = os.path.join(array_dir, "meta.json")
        fresh = False
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            fresh = all(meta.get(k) == v for k, v in wanted.items())
        if not fresh:
            log.info(f"Converting the dataset into arrays in {array_dir}")
            convert_to_arrays(
                self.dataset, array_dir, self.num_workers, neighbor_list=neighbor_list
            )
        self.data = DeviceAtomsData(array_dir, self.data_device)
        # padding pairs lie beyond the cutoff (all pairs: beyond any cutoff)
        self._pad_distance = (
            neighbor_list._cutoff + 1.0 if neighbor_list is not None else 1000.0
        )

        train_idx = np.asarray(self.train_idx, dtype=np.int64)
        seed = int(torch.randint(2**31, ()))  # follows seed_everything
        if self.batching == "budget":
            self.train_batches = BudgetBatches(
                self.data, train_idx, self.batch_size, seed
            )
            train_pad = self.train_batches.pad
        else:
            train_pad = self.data.pad_sizes(train_idx, self.batch_size, self.pad_k)
            self.train_batches = ShuffledBatches(
                self.data, train_idx, self.batch_size, train_pad, seed
            )
        self.pad = {"train": train_pad}
        for stage, idx, batch_size in (
            ("val", self.val_idx, self.val_batch_size),
            ("test", self.test_idx, self.test_batch_size),
        ):
            if len(idx):
                self.pad[stage] = self.data.exact_pad_sizes(idx, batch_size)
        self.provider = _ArrayStats(self.data, self.train_idx, self.dataset)

    def build_batch(self, sel: torch.Tensor, stage: str) -> dict[str, torch.Tensor]:
        """With data_device: the padded, transformed batch of the structures sel
        (``stage``: train, val or test). Fixed shapes and no host sync: recordable
        in a CUDA graph."""
        sel = sel.to(self.data.device, non_blocking=True)
        batch = self.data.padded_batch(sel, *self.pad[stage], self._pad_distance)
        for t in self._batch_transforms[stage]:
            batch = t(batch)
        return batch

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

    def train_dataloader(self):
        if self.data_device is not None:
            return _DeviceBatches(self, "train", self.train_batches, None)
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
        if self.data_device is not None:
            return _DeviceBatches(
                self,
                "val",
                OrderedBatches(self.val_idx, self.val_batch_size),
                self.val_seed,
            )
        if self._val_dataloader is None:
            self._val_dataloader = AtomsLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
            )

        return self._val_dataloader

    def test_dataloader(self):
        if self.data_device is not None:
            return _DeviceBatches(
                self,
                "test",
                OrderedBatches(self.test_idx, self.test_batch_size),
                self.val_seed,
            )
        if self._test_dataloader is None:
            self._test_dataloader = AtomsLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
            )

        return self._test_dataloader


def _split_transforms(transforms) -> tuple[Any, list]:
    """The neighbor list of a transform list (run at the conversion; None: all pairs)
    and the other transforms (run on batches on the device)."""
    pairs = [
        t
        for t in transforms
        if isinstance(t, (NeighborListTransform, AllToAllNeighborList))
    ]
    if len(pairs) > 1:
        raise ValueError(f"data_device takes one neighbor list, got {pairs}")
    rest = [t for t in transforms if not any(t is p for p in pairs)]
    for t in rest:
        if type(t).__module__ == NeighborListTransform.__module__:
            raise ValueError(f"{type(t).__name__} is not supported with data_device")
    if pairs and isinstance(pairs[0], NeighborListTransform):
        return pairs[0], rest
    return None, rest  # all pairs within each structure


class _ArrayStats:
    """Statistics of the train split from the arrays on the device, for the
    transforms and postprocessors: no second pass over the db (which took ~30-50 s
    for QM9), atom references fitted in float64."""

    def __init__(self, data: DeviceAtomsData, train_idx, dataset: ASEAtomsData):
        self.data, self.train_idx, self.dataset = data, train_idx, dataset

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atomrefs = self.dataset.atomrefs if remove_atomref else {}
        return self.data.stats(
            property, self.train_idx, divide_by_atoms, atomrefs.get(property)
        )

    def get_atomrefs(
        self, property: str, is_extensive: bool, estimate: bool = True
    ) -> dict[str, torch.Tensor]:
        given = self.dataset.atomrefs
        if property in given:
            return {property: given[property]}
        if not estimate:
            raise RuntimeError(
                f"The dataset provides no atomrefs for property '{property}' "
                "and atomref estimation is disabled."
            )
        return {property: self.data.atomrefs(property, self.train_idx, is_extensive)}


def _default_generator(device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return torch.cuda.default_generators[index]
    return torch.default_generator


class _DeviceBatches:
    """The batches of one stage with data_device, built on the device as they are
    drawn; or only their structure indices, when the training step builds the batch
    itself (AtomisticTask with cuda_graph).

    With a seed, the pass has a random state of its own, started from the seed: the
    device's default generator is switched to it while a batch is built and back
    right after, so random transforms (noise) draw the same values in every pass,
    and everything else continues its own sequence."""

    def __init__(self, datamodule, stage, indices, seed):
        self.dm, self.stage, self.indices, self.seed = datamodule, stage, indices, seed

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        build = not (self.stage == "train" and self.dm.train_batches_in_step)
        if self.seed is None:
            for sel in self.indices:
                yield self.dm.build_batch(sel, self.stage) if build else sel
            return
        gen = _default_generator(self.dm.data.device)
        outer = gen.get_state()
        own = gen.manual_seed(self.seed).get_state()
        gen.set_state(outer)
        for sel in self.indices:
            outer = gen.get_state()
            gen.set_state(own)
            try:
                batch = self.dm.build_batch(sel, self.stage)
                own = gen.get_state()
            finally:
                gen.set_state(outer)
            yield batch

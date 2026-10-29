"""
A LightningDataModule whose dataset lives on the GPU (:mod:`schnetpack.data.device`).
"""

import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data.device import (
    DeviceAtomsData,
    OrderedBatches,
    ShuffledBatches,
    convert_to_arrays,
    dataset_fingerprint,
)
from schnetpack.data.provider import StatsAtomrefProvider

__all__ = ["DeviceAtomsDataModule"]


def _default_generator(device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return torch.cuda.default_generators[index]
    return torch.default_generator


class _Batches:
    """Batches of one stage, built on the device by the datamodule as they are
    drawn; or only their structure indices, when the training step builds the
    batch itself (``GraphStepTask``).

    With a seed, the pass has a random state of its own, started from the seed:
    the device's default generator is switched to it while a batch is built and
    back right after, so random transforms (noise) draw the same values in every
    pass, and everything else continues its own sequence."""

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


class DeviceAtomsDataModule(pl.LightningDataModule):
    """
    A dataset held on a device, batches built and transformed there.

    The dataset is converted once into flat arrays (cached in ``cache_dir``), then
    held on ``device``. Every batch is padded to fixed sizes and holds all pairs
    within each structure (molecules); see :mod:`schnetpack.data.device`.
    Transforms run on whole batches on the device and must accept collated batches
    (e.g. ``SubtractCenterOfGeometry``, ``SubtractCenterOfMass``, ``CastTo32``,
    :class:`~schnetpack.generative.Diffuse`). Padding rows are marked by
    ``properties.atom_mask`` and ``properties.structure_mask``; losses must leave
    them out (:class:`~schnetpack.lightning.GraphStepTask` does).

    Training batches are random, of ``batch_size`` structures each (the last
    incomplete one dropped), padded to sizes that hold all but a ~1e-9 fraction
    of them (``pad_k`` standard deviations above the mean; batches that do not fit
    are skipped). Validation and test batches keep the order of their split and
    are padded to their largest batch. With ``val_seed``, random transforms draw
    the same values in every validation and test pass, so the losses compare
    across epochs.

    It serves statistics (``get_stats``, ``get_atomrefs``) like
    :class:`~schnetpack.lightning.AtomsDataModule`, from the dataset's train split.
    """

    def __init__(
        self,
        dataset: ASEAtomsData,
        batch_size: int,
        split_file: str,
        cutoff: float,
        transforms: Sequence[Any] | None = None,
        val_transforms: Sequence[Any] | None = None,
        test_transforms: Sequence[Any] | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        device: str | torch.device = "cuda",
        cache_dir: str | None = None,
        num_workers: int = 0,
        pad_k: float = 6.0,
        seed: int = 0,
        val_seed: int | None = 0,
        stats_file: str | None = None,
    ):
        """
        Args:
            dataset: the dataset (molecules), without transforms
            batch_size: structures per training batch
            split_file: npz file with ``train_idx``, ``val_idx`` and optionally
                ``test_idx``, indices into the dataset
            cutoff: the model's cutoff; padding pairs are placed beyond it
            transforms: batch transforms, applied in order to every batch
            val_transforms: for validation batches (default: ``transforms``)
            test_transforms: for test batches (default: ``val_transforms``)
            val_batch_size: structures per validation batch (default: batch_size)
            test_batch_size: per test batch (default: val_batch_size)
            device: where the data and the batches live
            cache_dir: where the converted arrays are kept (default:
                ``<db>.arrays`` next to the db); reused while the db, the
                dataset's properties and its units are unchanged
            num_workers: DataLoader workers for the one-time conversion
            pad_k: training pad sizes, standard deviations above the mean
            seed: seed of the training batch order
            val_seed: seed of random transforms in validation and testing (None:
                new draws every pass)
            stats_file: npz file persisting statistics (as AtomsDataModule's)
        """
        super().__init__()
        self.dataset, self.batch_size, self.split_file = dataset, batch_size, split_file
        self.cutoff = cutoff
        self.transforms = {
            "train": list(transforms or []),
            "val": list(
                val_transforms if val_transforms is not None else transforms or []
            ),
        }
        self.transforms["test"] = (
            list(test_transforms)
            if test_transforms is not None
            else self.transforms["val"]
        )
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size
        self.device = torch.device(device)
        self.cache_dir = cache_dir or f"{dataset.datapath}.arrays"
        self.num_workers, self.pad_k = num_workers, pad_k
        self.seed, self.val_seed = seed, val_seed
        self.stats_file = stats_file
        self.data: DeviceAtomsData | None = None
        self.provider: StatsAtomrefProvider | None = None
        # set by a training step that builds its own batches from indices
        self.train_batches_in_step = False

    def setup(self, stage: str | None = None):
        if self.data is not None:
            return
        meta_file = os.path.join(self.cache_dir, "meta.json")
        source = json.loads(json.dumps(dataset_fingerprint(self.dataset)))
        fresh = False
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                fresh = json.load(f).get("source") == source
        if not fresh:
            convert_to_arrays(self.dataset, self.cache_dir, self.num_workers)
        self.data = DeviceAtomsData(self.cache_dir, self.device)
        split = np.load(self.split_file)
        self.train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        self.val_idx = np.asarray(split["val_idx"], dtype=np.int64)
        self.test_idx = (
            np.asarray(split["test_idx"], dtype=np.int64)
            if "test_idx" in split
            else None
        )
        self.pad = {
            "train": self.data.pad_sizes(self.train_idx, self.batch_size, self.pad_k),
            "val": self.data.exact_pad_sizes(self.val_idx, self.val_batch_size),
        }
        if self.test_idx is not None:
            self.pad["test"] = self.data.exact_pad_sizes(
                self.test_idx, self.test_batch_size
            )
        self.provider = StatsAtomrefProvider(
            self.dataset, self.train_idx.tolist(), self.stats_file
        )
        unique = {id(t): t for ts in self.transforms.values() for t in ts}
        for t in unique.values():
            t.initialize(self)

    def build_batch(self, sel: torch.Tensor, stage: str) -> dict[str, torch.Tensor]:
        """The padded, transformed batch of the structures sel (``stage``: train,
        val or test). Fixed shapes and no host sync: recordable in a CUDA graph."""
        sel = sel.to(self.device, non_blocking=True)
        batch = self.data.padded_batch(sel, *self.pad[stage], cutoff=self.cutoff)
        for t in self.transforms[stage]:
            batch = t(batch)
        return batch

    def train_dataloader(self):
        self.train_batches = ShuffledBatches(
            self.data, self.train_idx, self.batch_size, self.pad["train"], self.seed
        )
        return _Batches(self, "train", self.train_batches, None)

    def val_dataloader(self):
        return _Batches(
            self,
            "val",
            OrderedBatches(self.val_idx, self.val_batch_size),
            self.val_seed,
        )

    def test_dataloader(self):
        if self.test_idx is None:
            raise ValueError(f"{self.split_file} has no test_idx")
        return _Batches(
            self,
            "test",
            OrderedBatches(self.test_idx, self.test_batch_size),
            self.val_seed,
        )

    # the stats-source protocol of AtomsDataModule (transforms, AddOffsets)
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

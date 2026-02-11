from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Union

import torch

from schnetpack.data import AtomsDataModule, calculate_stats, estimate_atomrefs
from schnetpack.datasets.factories import build_dataset_by_name
from schnetpack.datasets.merged_dataset import MergedDataset, SplitSize


__all__ = ["MergedAtomsDataModule"]


class MergedAtomsDataModule(AtomsDataModule):
    """
    A LightningDataModule that:
      - prepares component datasets (download/create component DBs if missing)
      - builds a virtual merged dataset at runtime (no merged DB saved)
      - provides balanced train/val/test splits according to proportions
      - injects dataset_id and source_index via MergedDataset
    """

    def __init__(
        self,
        dataset_names: List[str],
        proportions: Dict[str, float],
        molecule: str,
        dataset_root: str,
        total_size: int,
        batch_size: int,
        num_train: SplitSize,
        num_val: SplitSize,
        num_test: Optional[SplitSize] = None,
        seed: int = 42,
        load_properties: Optional[List[str]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        # We call parent init for consistent structure.
        # datapath is unused because we don't load a merged DB.
        super().__init__(
            datapath=f"{dataset_root}/_virtual_merged.db",
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=None,
            format=None,
            load_properties=load_properties,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

        self.dataset_names = dataset_names
        self.proportions = proportions
        self.molecule = molecule
        self.dataset_root = dataset_root
        self.total_size = int(total_size)
        self.seed = int(seed)

        self.component_datasets: Optional[Dict[str, Any]] = None
        self.merged_atomrefs: Optional[Dict[str, torch.Tensor]] = None

        # override parent internals (we won't use parent splitting)
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.dataset = None

    @staticmethod
    def _atomrefs_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """
        Strict equality check for atomrefs dict (keys + tensor values).
        """
        if a.keys() != b.keys():
            return False
        for k in a.keys():
            va, vb = a[k], b[k]
            if torch.is_tensor(va) and torch.is_tensor(vb):
                if va.shape != vb.shape:
                    return False
                if not torch.allclose(va, vb):
                    return False
            else:
                if va != vb:
                    return False
        return True

    def _compute_merged_atomrefs(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Only keep atomrefs if all component datasets provide them and they are identical.
        Otherwise return None to avoid wrong subtraction.
        """
        assert self.component_datasets is not None

        refs = []
        for _, ds in self.component_datasets.items():
            refs.append(getattr(ds, "atomrefs", None))

        if any(r is None for r in refs):
            return None

        base = refs[0]
        for r in refs[1:]:
            if not self._atomrefs_equal(base, r):
                return None

        return base

    def prepare_data(self):
        """
        Ensure component datasets exist on disk and load them.
        """
        self.component_datasets = {}
        for name in self.dataset_names:
            self.component_datasets[name] = build_dataset_by_name(
                name=name,
                molecule=self.molecule,
                dataset_root=self.dataset_root,
                load_properties=self.load_properties,
            )

        # compute atomrefs if consistent
        self.merged_atomrefs = self._compute_merged_atomrefs()

    def setup(self, stage: Optional[str] = None):
        """
        Build runtime train/val/test merged datasets and attach transforms.
        """
        if self.component_datasets is None:
            self.prepare_data()

        train_ds, val_ds, test_ds = MergedDataset.make_splits(
            datasets=self.component_datasets,
            proportions=self.proportions,
            total_size=self.total_size,
            num_train=self.num_train,
            num_val=self.num_val,
            num_test=self.num_test,
            seed=self.seed,
            shuffle_within_split=True,
            add_source_index=True,
            atomrefs=self.merged_atomrefs,
        )

        # Make SchNetPack code happy: expose atomrefs attribute
        train_ds.atomrefs = self.merged_atomrefs
        val_ds.atomrefs = self.merged_atomrefs
        test_ds.atomrefs = self.merged_atomrefs

        self._train_dataset = train_ds
        self._val_dataset = val_ds
        self._test_dataset = test_ds

        # attach transforms using the parent helper
        self._setup_transforms()

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean/std on merged training data.

        If remove_atomref=True but merged atomrefs are not available/compatible,
        we silently disable atomref subtraction.
        """
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats:
            return self._stats[key]

        atomref = None
        if (
            remove_atomref
            and self.merged_atomrefs is not None
            and property in self.merged_atomrefs
        ):
            atomref = self.merged_atomrefs

        stats = calculate_stats(
            self.train_dataloader(),
            divide_by_atoms={property: divide_by_atoms},
            atomref=atomref,
        )[property]

        self._stats[key] = stats
        return stats

    def get_atomrefs(
        self, property: str, is_extensive: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate atomrefs from merged training data (optional utility).
        """
        key = (property, is_extensive)
        if key in self._atomrefs:
            return {property: self._atomrefs[key]}

        atomrefs = estimate_atomrefs(
            self.train_dataloader(),
            is_extensive={property: is_extensive},
        )[property]

        self._atomrefs[key] = atomrefs
        return {property: atomrefs}

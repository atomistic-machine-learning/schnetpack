"""
Assumptions: ## NOTE: ASK STEFAAN
    - All component datasets share the same properties and units.
    - Atomrefs are identical across all components (or all None).
    - Distance units agree across all components.

Usage
-----
    md17  = MD17(datapath="data/md17_aspirin.db", molecule="aspirin")
    rmd17 = RMD17(datapath="data/rmd17_aspirin.db", molecule="aspirin")

    merged = MergedDataset(
        datasets={"md17": md17, "rmd17": rmd17},
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.RemoveOffsets(property="energy", remove_mean=True),
            trn.CastTo32(),
        ],
    )

    dm = AtomsDataModuleV2(
        dataset=merged,
        batch_size=32,
        num_train=0.8,
        num_val=0.1,
        splitting=ProportionalSplit({"md17": 0.5, "rmd17": 0.5}),
        split_file="split.npz",
    )

    dm.setup() 
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from schnetpack.data.atoms import ASEAtomsData, AtomsDataError
from schnetpack.transform.base import Transform

__all__ = ["MergedDataset"]

class MergedDataset(Dataset):
    """
    Virtual merged dataset.

    What is NOT duplicated from ASEAtomsData (no backing DB here):
        metadata, _set_metadata, update_metadata, _check_db,
        _get_properties, create, download, add_system, add_systems.

    What IS reimplemented (plan-based, not subset_idx / conn based):
        __len__, __getitem__, subset, load_properties, _apply_transforms,
        and read-only properties that delegate to the first component dataset.

    Transform ownership ## NOTE: ASK STEFAAN (BIGGEST DOUBT)
    -------------------
    Transforms are passed to MergedDataset directly and applied once per
    sample after fetching from the component dataset.  Component datasets
    must NOT carry their own transforms to avoid double-application.
    Transforms that require statistics (e.g. RemoveOffsets) are initialized
    by AtomsDataModuleV2.setup() via StatsAtomrefProvider on the train split.

    Per-sample fields added vs a plain ASEAtomsData sample:
        dataset_id   (int64, shape [1]) — insertion-order id per component
        source_index (int64, shape [1]) — original index in component dataset
    """

    def __init__(
        self,
        datasets: Dict[str, ASEAtomsData],
        transforms: Optional[List[Transform]] = None,
        train_transforms: Optional[List[Transform]] = None,
        val_transforms: Optional[List[Transform]] = None,
        test_transforms: Optional[List[Transform]] = None,
        add_source_index: bool = True,
    ) -> None:
        """
        Args:
            datasets: component datasets keyed by an arbitrary name.
                      Insertion order determines dataset_id (0, 1, 2, …).
            transforms: shared transforms applied to every sample.
            train_transforms: overrides transforms for the train split.
            val_transforms: overrides transforms for the val split.
            test_transforms: overrides transforms for the test split.
            add_source_index: inject ``source_index`` into each sample.
        """
        if not datasets:
            raise AtomsDataError("datasets must not be empty.")

        self._validate_compatibility(datasets)
        self._warn_if_component_transforms(datasets)

        self.datasets = datasets
        self.add_source_index = add_source_index

        # Integer id per dataset name, in insertion order
        self._dataset_ids: Dict[str, int] = {
            name: i for i, name in enumerate(datasets)
        }

        self.plan: List[Tuple[str, int]] = [
                            (name, idx)
                            for name, ds in datasets.items()
                            for idx in range(len(ds))
                        ]

        self.transforms: List[Transform] = list(transforms or [])
        self.train_transforms: Optional[List[Transform]] = (
            list(train_transforms) if train_transforms else None
        )
        self.val_transforms: Optional[List[Transform]] = (
            list(val_transforms) if val_transforms else None
        )
        self.test_transforms: Optional[List[Transform]] = (
            list(test_transforms) if test_transforms else None
        )

        # Set by subset() — drives split-aware transform dispatch
        self.split: Optional[str] = None

        self._load_properties: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Compatibility validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_compatibility(datasets: Dict[str, ASEAtomsData]) -> None:
        """
        Enforce that all component datasets share:
            - the same distance unit
            - the same property names and units
            - identical atomrefs (or all None)
        Raises AtomsDataError with a descriptive message on any mismatch.
        """
        names = list(datasets.keys())
        ref_name = names[0]
        ref = datasets[ref_name]

        for name in names[1:]:
            ds = datasets[name]

            if ds.distance_unit != ref.distance_unit:
                raise AtomsDataError(
                    f"Distance unit mismatch: '{ref_name}' has "
                    f"'{ref.distance_unit}' but '{name}' has '{ds.distance_unit}'."
                )

            if ds.units != ref.units:
                raise AtomsDataError(
                    f"Property unit mismatch between '{ref_name}' and '{name}'.\n"
                    f"  {ref_name}: {ref.units}\n"
                    f"  {name}: {ds.units}"
                )

        all_atomrefs = [getattr(ds, "atomrefs", None) for ds in datasets.values()]
        has_atomrefs = [r is not None for r in all_atomrefs]

        if any(has_atomrefs) and not all(has_atomrefs):
            missing = [n for n, h in zip(names, has_atomrefs) if not h]
            raise AtomsDataError(
                f"Some datasets have atomrefs and some do not. "
                f"Datasets missing atomrefs: {missing}"
            )

        if all(has_atomrefs):
            ref_ar = all_atomrefs[0]
            for name, ar in zip(names[1:], all_atomrefs[1:]):
                if ar.keys() != ref_ar.keys():
                    raise AtomsDataError(
                        f"Atomref property keys differ between "
                        f"'{ref_name}' and '{name}'."
                    )
                for prop in ref_ar:
                    if not torch.allclose(
                        torch.tensor(ref_ar[prop], dtype=torch.float64),
                        torch.tensor(ar[prop], dtype=torch.float64),
                    ):
                        raise AtomsDataError(
                            f"Atomref values for '{prop}' differ between "
                            f"'{ref_name}' and '{name}'."
                        )

    @staticmethod
    def _warn_if_component_transforms(datasets: Dict[str, ASEAtomsData]) -> None:
        """
        Warn if any component dataset already has transforms set.
        Transforms should live on MergedDataset, not on components,
        to avoid double-application.
        """
        import warnings
        for name, ds in datasets.items():
            if getattr(ds, "transforms", None):
                warnings.warn(
                    f"Component dataset '{name}' has transforms set. "
                    f"These will be bypassed by MergedDataset.__getitem__ "
                    f"to avoid double-application. Move all transforms to "
                    f"MergedDataset instead.",
                    UserWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Read-only properties — delegate to first component dataset.
    # All components are guaranteed identical by _validate_compatibility.
    # ------------------------------------------------------------------

    @property
    def available_properties(self) -> List[str]:
        return next(iter(self.datasets.values())).available_properties

    @property
    def units(self) -> Dict[str, str]:
        return next(iter(self.datasets.values())).units

    @property
    def distance_unit(self) -> str:
        return next(iter(self.datasets.values())).distance_unit

    @property
    def atomrefs(self) -> Optional[Dict]:
        return getattr(next(iter(self.datasets.values())), "atomrefs", None)

    # ------------------------------------------------------------------
    # load_properties — reimplemented (no backing DB conn)
    # Same contract as ASEAtomsData.load_properties
    # ------------------------------------------------------------------

    @property
    def load_properties(self) -> List[str]:
        if self._load_properties is None:
            return self.available_properties
        return self._load_properties

    @load_properties.setter
    def load_properties(self, val: Optional[List[str]]) -> None:
        if val is not None:
            missing = [p for p in val if p not in self.available_properties]
            if missing:
                raise AtomsDataError(
                    f"Properties not available in merged dataset: {missing}"
                )
        self._load_properties = val

    # ------------------------------------------------------------------
    # subset() — reimplemented to slice self.plan instead of subset_idx.
    # Called by AtomsDataModuleV2.setup() to build train/val/test views.
    # ------------------------------------------------------------------

    def subset(
        self,
        subset_idx: List[int],
        split: Optional[str] = None,
    ) -> "MergedDataset":
        ds = copy.copy(self)
        ds.plan = [self.plan[i] for i in subset_idx]
        ds.split = split
        return ds

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plan)
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        dataset_name, index = self.plan[i]
        component_ds = self.datasets[dataset_name]

        saved_transforms = component_ds.transforms
        saved_split = component_ds.split
        component_ds.transforms = []
        component_ds.split = None
        try:
            sample = component_ds[index]
        finally:
            component_ds.transforms = saved_transforms
            component_ds.split = saved_split

        sample["dataset_id"] = torch.tensor(
            [self._dataset_ids[dataset_name]], dtype=torch.long
        )
        if self.add_source_index:
            sample["source_index"] = torch.tensor([index], dtype=torch.long)

        return self._apply_transforms(sample)

    def _apply_transforms(
        self, props: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.split == "train" and self.train_transforms is not None:
            transforms = self.train_transforms
        elif self.split == "val" and self.val_transforms is not None:
            transforms = self.val_transforms
        elif self.split == "test" and self.test_transforms is not None:
            transforms = self.test_transforms
        else:
            transforms = self.transforms

        for tf in transforms:
            props = tf(props)
        return props
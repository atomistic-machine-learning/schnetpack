from typing import Dict, List, Optional
import warnings

import torch
from ase.data import atomic_masses

import schnetpack.properties as structure
from .base import Transform
from schnetpack.nn import scatter_add
from schnetpack.data.provider import StatsAtomrefProvider


__all__ = [
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
    "AddOffsets",
    "RemoveOffsets",
    "ScaleProperty",
    "ConditionalAddOffsets",
    "ConditionalRemoveOffsets",
]


class SubtractCenterOfMass(Transform):
    """
    Subtract center of mass from positions.

    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        masses = torch.tensor(atomic_masses[inputs[structure.Z]])
        inputs[structure.position] -= (
            masses.unsqueeze(-1) * inputs[structure.position]
        ).sum(0) / masses.sum()
        return inputs


class SubtractCenterOfGeometry(Transform):
    """
    Subtract center of geometry from positions.

    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs[structure.position] -= inputs[structure.position].mean(0)
        return inputs


class RemoveOffsets(Transform):
    """
    Remove offsets from property based on the mean of the training data and/or the
    single atom reference calculations.

    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(
        self,
        property,
        remove_mean: bool = False,
        remove_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
        atomrefs: torch.Tensor = None,
        property_mean: torch.Tensor = None,
        estimate_atomref: bool = False,
    ):
        """
        Args:
            property: The property to add the offsets to.
            remove_mean: If true, remove mean of the dataset from property.
            remove_atomrefs: If true, remove single-atom references.
            is_extensive: Set true if the property is extensive.
            zmax: Set the maximum atomic number, to determine the size of the atomref
                tensor.
            atomrefs: Provide single-atom references directly.
            property_mean: Provide mean property value / n_atoms.
            estimate_atomref: If true, add estimated atomrefs.
        """
        super().__init__()
        self._property = property
        self.remove_mean = remove_mean
        self.remove_atomrefs = remove_atomrefs
        self.is_extensive = is_extensive
        self.estimate_atomref = estimate_atomref

        assert not (
            estimate_atomref and atomrefs is not None
        ), "You can not set `atomrefs` and use `estimate_atomrefs=True!`"

        if atomrefs is not None:
            self._atomrefs_initialized = True
        else:
            self._atomrefs_initialized = False

        if property_mean is not None:
            self._mean_initialized = True
        else:
            self._mean_initialized = False

        if self.remove_atomrefs:
            # atomrefs = atomrefs or torch.zeros((zmax,)) #NOTE: this was the original code
            atomrefs = atomrefs if atomrefs is not None else torch.zeros((zmax,))
            self.register_buffer("atomref", atomrefs)
        if self.remove_mean:
            property_mean = property_mean or torch.zeros((1,))
            self.register_buffer("mean", property_mean)

    def initialize(self, provider, atomrefs=None, **kwargs) -> None:
        """
        Initialize mean and/or atomref using a StatsAtomrefProvider.
        """
        if self.remove_atomrefs and not self._atomrefs_initialized:
            if self.estimate_atomref:
                atrefs = provider.get_atomrefs(self._property, self.is_extensive)
            else:
                if atomrefs is None:
                    raise RuntimeError(
                        "RemoveOffsets requires dataset atomrefs when estimate_atomref=False."
                    )
                atrefs = atomrefs
            self.atomref = atrefs[self._property].detach()

        if self.remove_mean and not self._mean_initialized:
            mean, _std = provider.get_stats(
                self._property, self.is_extensive, self.remove_atomrefs
            )
            self.mean = mean.detach()

    # legacy hook for old AtomsDataModule
    def datamodule(self, _datamodule):
        """
        Legacy hook for old AtomsDataModule. Safe to remove once legacy DM is removed.
        """
        warnings.warn(
            "RemoveOffsets.datamodule(...) is deprecated and will be removed in a future "
            "release. Use initialize(provider=..., atomrefs=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        provider = StatsAtomrefProvider(_datamodule.train_dataset)
        return self.initialize(provider, atomrefs=provider.train_atomrefs)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.remove_mean:
            mean = (
                self.mean * inputs[structure.n_atoms]
                if self.is_extensive
                else self.mean
            )
            inputs[self._property] -= mean

        if self.remove_atomrefs:
            atomref_bias = torch.sum(self.atomref[inputs[structure.Z]])
            if not self.is_extensive:
                atomref_bias = atomref_bias / inputs[structure.n_atoms]
            inputs[self._property] -= atomref_bias
        return inputs


class ScaleProperty(Transform):
    """
    Scale an entry of the input or results dioctionary.

    The `scale` can be automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, it has to be provided in the init manually.

    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(
        self,
        input_key: str,
        target_key: str = None,
        output_key: str = None,
        scale_by_mean: bool = False,
        scale: torch.Tensor = None,
    ):
        """
        Args:
            input_key: dict key of input to be scaled
            target_key: dict key of target to derive scaling from
                (either its mean or std dev)
            output_key: dict key for scaled output
            scale_by_mean: if true, use the mean of the target variable for scaling,
                otherwise use its standard deviation
            scale: provide the scale of the property manually.
        """
        super().__init__()
        self.input_key = input_key
        self._target_key = target_key or input_key
        self.output_key = output_key or input_key
        self._scale_by_mean = scale_by_mean
        self.model_outputs = [self.output_key]

        if scale is not None:
            self._initialized = True
        else:
            self._initialized = False

        scale = scale or torch.ones((1,))
        self.register_buffer("scale", scale)

    def initialize(self, provider, atomrefs=None) -> None:
        """
        Initialize scaling using training statistics.
        """
        if not self._initialized:
            mean, std = provider.get_stats(self._target_key, True, False)
            scale = mean if self._scale_by_mean else std
            self.scale = torch.abs(scale).detach()

    def datamodule(self, _datamodule):
        """
        Legacy hook for old AtomsDataModule. Safe to remove once legacy DM is removed.
        """
        warnings.warn(
            "ScaleProperty.datamodule(...) is deprecated and will be removed in a future "
            "release. Use initialize(provider=..., atomrefs=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        provider = StatsAtomrefProvider(_datamodule.train_dataset)
        return self.initialize(provider, atomrefs=None)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs[self.output_key] = inputs[self.input_key] * self.scale
        return inputs


class AddOffsets(Transform):
    """
    Add offsets to property based on the mean of the training data and/or the single
    atom reference calculations.

    """

    is_preprocessor: bool = False
    is_postprocessor: bool = True
    atomref: torch.Tensor

    def __init__(
        self,
        property,
        add_mean: bool = False,
        add_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
        atomrefs: torch.Tensor = None,
        property_mean: torch.Tensor = None,
        estimate_atomref: bool = False,
    ):
        """
        Args:
            property: The property to add the offsets to.
            add_mean: If true, add mean of the dataset.
            add_atomrefs: If true, add single-atom references.
            is_extensive: Set true if the property is extensive.
            zmax: Set the maximum atomic number, to determine the size of the atomref
                tensor.
            atomrefs: Provide single-atom references directly.
            property_mean: Provide mean property value / n_atoms.
            estimate_atomref: If true, add estimated atomrefs.
        """
        super().__init__()
        self._property = property
        self.add_mean = add_mean
        self.add_atomrefs = add_atomrefs
        self.is_extensive = is_extensive
        self._aggregation = "sum" if self.is_extensive else "mean"
        self.estimate_atomref = estimate_atomref

        assert not (
            estimate_atomref and atomrefs is not None
        ), "You can not set `atomrefs` and use `estimate_atomrefs=True!`"

        if atomrefs is not None:
            self._atomrefs_initialized = True
        else:
            self._atomrefs_initialized = False

        if property_mean is not None:
            self._mean_initialized = True
        else:
            self._mean_initialized = False

        # atomrefs = atomrefs or torch.zeros((zmax,)) #NOTE: this was the original code
        atomrefs = atomrefs if atomrefs is not None else torch.zeros((zmax,))
        property_mean = property_mean or torch.zeros((1,))
        self.register_buffer("atomref", atomrefs)
        self.register_buffer("mean", property_mean)

    def initialize(self, provider, atomrefs=None) -> None:
        """
        Initialize mean and/or atomref using a StatsAtomrefProvider.
        """
        if self.add_atomrefs and not self._atomrefs_initialized:
            if self.estimate_atomref:
                atrefs = provider.get_atomrefs(self._property, self.is_extensive)
            else:
                if atomrefs is None:
                    raise RuntimeError(
                        "AddOffsets requires dataset atomrefs when estimate_atomref=False."
                    )
                atrefs = atomrefs
            self.atomref = atrefs[self._property].detach()

        if self.add_mean and not self._mean_initialized:
            mean, _std = provider.get_stats(
                self._property, self.is_extensive, self.add_atomrefs
            )
            self.mean = mean.detach()

    def datamodule(self, _datamodule):
        """
        Legacy hook for old AtomsDataModule. Safe to remove once legacy DM is removed.
        """
        warnings.warn(
            "AddOffsets.datamodule(...) is deprecated and will be removed in a future "
            "release. Use initialize(provider=..., atomrefs=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        provider = StatsAtomrefProvider(_datamodule.train_dataset)
        return self.initialize(provider, atomrefs=provider.train_atomrefs)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.add_mean:
            mean = (
                self.mean * inputs[structure.n_atoms]
                if self.is_extensive
                else self.mean
            )
            inputs[self._property] += mean

        if self.add_atomrefs:
            idx_m = inputs[structure.idx_m]
            y0i = self.atomref[inputs[structure.Z]]
            maxm = int(idx_m[-1]) + 1

            y0 = scatter_add(y0i, idx_m, dim_size=maxm)

            if not self.is_extensive:
                y0 /= inputs[structure.n_atoms]

            inputs[self._property] += y0

        return inputs


class ConditionalAddOffsets(Transform):
    """
    Per-dataset variant of AddOffsets for use with MergedDataset.
    """

    is_preprocessor: bool = False
    is_postprocessor: bool = True

    SOURCE_INDEX_KEY = "source_index"

    def __init__(
        self,
        property: str,
        dataset_names: List[str],
        add_mean: bool = False,
        add_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
        estimate_atomref: bool = False,
    ):
        """
        Args:
            property: The property to add the offsets to.
            dataset_names: List of dataset names.
            add_mean: If true, add mean of the dataset.
            add_atomrefs: If true, add single-atom references.
            is_extensive: Set true if the property is extensive.
            zmax: Set the maximum atomic number, to determine the size of the atomref
                tensor.
            estimate_atomref: If true, add estimated atomrefs.
        """
        super().__init__()
        self._property = property
        self.dataset_names = dataset_names  # index 0 → name[0], etc.
        self.add_mean = add_mean
        self.add_atomrefs = add_atomrefs
        self.is_extensive = is_extensive
        self.estimate_atomref = estimate_atomref
        self._n = len(dataset_names)

        # One mean scalar and one atomref vector per dataset.
        #   means    : [n_datasets, 1]
        #   atomrefs : [n_datasets, zmax]
        self.register_buffer("means", torch.zeros(self._n, 1))
        self.register_buffer("atomrefs", torch.zeros(self._n, zmax))

        self._mean_initialized = [False] * self._n
        self._atomrefs_initialized = [False] * self._n

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, provider, atomrefs=None) -> None:
        """
        Populate per-dataset buffers from a MergedStatsAtomrefProvider.
        """
        if not hasattr(provider, "providers"):
            raise TypeError(
                "ConditionalAddOffsets requires a MergedStatsAtomrefProvider "
                f"(got {type(provider).__name__}). "
                "Make sure your datamodule uses provider: MergedStatsAtomrefProvider."
            )

        for idx, name in enumerate(self.dataset_names):
            component_provider = provider.providers[name]

            if self.add_mean and not self._mean_initialized[idx]:
                mean, _ = component_provider.get_stats(
                    self._property, self.is_extensive, self.add_atomrefs
                )
                self.means[idx] = mean.detach()
                self._mean_initialized[idx] = True

            if self.add_atomrefs and not self._atomrefs_initialized[idx]:
                if self.estimate_atomref:
                    atrefs = component_provider.get_atomrefs(
                        self._property, self.is_extensive
                    )
                else:
                    # fall back to precomputed train atomrefs stored on the provider
                    train_ar = provider.train_atomrefs.get(name)
                    if train_ar is None or self._property not in train_ar:
                        raise RuntimeError(
                            f"No precomputed atomrefs for dataset '{name}', "
                            f"property '{self._property}'. "
                            "Set estimate_atomref=True or supply atomrefs manually."
                        )
                    atrefs = train_ar

                ar_tensor = atrefs[self._property].detach()
                length = ar_tensor.shape[0]
                self.atomrefs[idx, :length] = ar_tensor
                self._atomrefs_initialized[idx] = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add per-sample offsets back to ``inputs[self._property]``.

        Batch layout
        ------------
        inputs[SOURCE_INDEX_KEY] : [batch_size]  int, one entry per molecule
        inputs[structure.idx_m]  : [n_atoms]     int, maps atom → molecule
        inputs[structure.n_atoms]: [batch_size]  int
        inputs[structure.Z]      : [n_atoms]     int, atomic numbers
        """
        source_idx = inputs[self.SOURCE_INDEX_KEY]  # [batch_size]
        idx_m = inputs[structure.idx_m]  # [n_atoms]
        n_atoms = inputs[structure.n_atoms]  # [batch_size]

        if self.add_mean:
            # means[source_idx] → [batch_size, 1] → squeeze → [batch_size]
            per_mol_mean = self.means[source_idx].squeeze(-1)  # [batch_size]
            if self.is_extensive:
                per_mol_mean = per_mol_mean * n_atoms  # scale by size
            inputs[self._property] += per_mol_mean

        if self.add_atomrefs:
            # Build a per-atom atomref lookup:
            #   dataset_idx_per_atom : which dataset each atom belongs to
            #   Z_per_atom           : atomic number of each atom
            dataset_idx_per_atom = source_idx[idx_m]  # [n_atoms]
            Z = inputs[structure.Z]  # [n_atoms]

            # atomrefs[dataset_idx_per_atom, Z] → [n_atoms]
            per_atom_ref = self.atomrefs[dataset_idx_per_atom, Z]  # [n_atoms]

            # Scatter-sum over molecules → [batch_size]
            batch_size = int(idx_m[-1]) + 1
            per_mol_ref = scatter_add(per_atom_ref, idx_m, dim_size=batch_size)

            if not self.is_extensive:
                per_mol_ref = per_mol_ref / n_atoms

            inputs[self._property] += per_mol_ref

        return inputs


class ConditionalRemoveOffsets(Transform):
    """
    Per-dataset variant of RemoveOffsets for use with MergedDataset.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    SOURCE_INDEX_KEY = "source_index"

    def __init__(
        self,
        property: str,
        dataset_names: List[str],
        remove_mean: bool = False,
        remove_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
        estimate_atomref: bool = False,
    ):
        """
        Args:
            property: The property to add the offsets to.
            dataset_names: List of dataset names.
            add_mean: If true, add mean of the dataset.
            add_atomrefs: If true, add single-atom references.
            is_extensive: Set true if the property is extensive.
            zmax: Set the maximum atomic number, to determine the size of the atomref
                tensor.
            estimate_atomref: If true, add estimated atomrefs.
        """
        super().__init__()
        self._property = property
        self.dataset_names = dataset_names
        self.remove_mean = remove_mean
        self.remove_atomrefs = remove_atomrefs
        self.is_extensive = is_extensive
        self.estimate_atomref = estimate_atomref
        self._n = len(dataset_names)

        # Stacked buffers — same layout as ConditionalAddOffsets
        # means    : [n_datasets, 1]
        # atomrefs : [n_datasets, zmax]
        self.register_buffer("means", torch.zeros(self._n, 1))
        self.register_buffer("atomrefs", torch.zeros(self._n, zmax))

        self._mean_initialized = [False] * self._n
        self._atomrefs_initialized = [False] * self._n

    def initialize(self, provider, atomrefs=None) -> None:
        """
        Populate per-dataset buffers from a MergedStatsAtomrefProvider.
        """
        if not hasattr(provider, "providers"):
            raise TypeError(
                "ConditionalRemoveOffsets requires a MergedStatsAtomrefProvider "
                f"(got {type(provider).__name__}). "
                "Make sure your datamodule uses provider: MergedStatsAtomrefProvider."
            )

        for idx, name in enumerate(self.dataset_names):
            component_provider = provider.providers[name]

            if self.remove_mean and not self._mean_initialized[idx]:
                mean, _ = component_provider.get_stats(
                    self._property, self.is_extensive, self.remove_atomrefs
                )
                self.means[idx] = mean.detach()
                self._mean_initialized[idx] = True

            if self.remove_atomrefs and not self._atomrefs_initialized[idx]:
                if self.estimate_atomref:
                    atrefs = component_provider.get_atomrefs(
                        self._property, self.is_extensive
                    )
                else:
                    train_ar = provider.train_atomrefs.get(name)
                    if train_ar is None or self._property not in train_ar:
                        raise RuntimeError(
                            f"No precomputed atomrefs for dataset '{name}', "
                            f"property '{self._property}'. "
                            "Set estimate_atomref=True or supply atomrefs manually."
                        )
                    atrefs = train_ar

                ar_tensor = atrefs[self._property].detach()
                length = ar_tensor.shape[0]
                self.atomrefs[idx, :length] = ar_tensor
                self._atomrefs_initialized[idx] = True

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source_idx = inputs[self.SOURCE_INDEX_KEY].view(-1).long()
        n_atoms = inputs[structure.n_atoms].view(-1)

        if self.remove_mean:
            per_mol_mean = self.means[source_idx].squeeze(-1)

            if self.is_extensive:
                per_mol_mean = per_mol_mean * n_atoms

            inputs[self._property] -= per_mol_mean

        if self.remove_atomrefs:
            idx_m = inputs[structure.idx_m]
            dataset_idx_per_atom = source_idx[idx_m]
            Z = inputs[structure.Z]

            per_atom_ref = self.atomrefs[dataset_idx_per_atom, Z]
            batch_size = int(idx_m[-1]) + 1
            per_mol_ref = scatter_add(per_atom_ref, idx_m, dim_size=batch_size)

            if not self.is_extensive:
                per_mol_ref = per_mol_ref / n_atoms

            inputs[self._property] -= per_mol_ref

        return inputs

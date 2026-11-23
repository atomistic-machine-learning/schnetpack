import torch
from ase.data import atomic_masses

import schnetpack.properties as structure
from schnetpack.model_format import register
from schnetpack.nn import scatter_add
from schnetpack.utils.cell import (
    atom_cells,
    niggli_reduce,
    structure_index,
    to_cart,
    to_frac,
)

from .base import Transform

__all__ = [
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
    "NiggliReduce",
    "WrapPositions",
    "AddOffsets",
    "RemoveOffsets",
    "ScaleProperty",
]


def _segment_mean(
    x: torch.Tensor, weight: torch.Tensor, idx_m: torch.Tensor, n_structures: int
) -> torch.Tensor:
    """Weighted mean of the rows of x per structure (idx_m), no host sync."""
    total = weight.new_zeros(n_structures).index_add_(0, idx_m, weight)
    sums = x.new_zeros(n_structures, x.shape[1]).index_add_(
        0, idx_m, weight[:, None] * x
    )
    return sums / total.clamp(min=torch.finfo(total.dtype).tiny)[:, None]


def _fit_atomrefs(atomrefs: torch.Tensor, size: int) -> torch.Tensor:
    """``atomrefs`` as a table of ``size`` entries, the size the transform was built
    with (``zmax``) and that a model built from its hyperparameters has: a shorter
    table is padded with zeros, and entries from ``size`` on must be zero."""
    atomrefs = atomrefs.detach()
    if atomrefs[size:].any():
        raise ValueError(
            f"the atomrefs have entries for atomic numbers from {size} on, beyond "
            f"the transform's table of {size} entries: build it with a larger zmax"
        )
    atomrefs = atomrefs[:size]
    return torch.nn.functional.pad(atomrefs, (0, size - len(atomrefs)))


@register
class SubtractCenterOfMass(Transform):
    """
    Subtract center of mass from positions.

    Works on one structure, or on a collated batch (``idx_m`` present, e.g. the
    padded batches of :class:`~schnetpack.lightning.AtomsDataModule` with
    ``cache``), structure by structure on the batch's device.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self):
        super().__init__()
        self._masses: dict[tuple, torch.Tensor] = {}

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        R = inputs[structure.position]
        if structure.idx_m not in inputs:
            masses = torch.tensor(atomic_masses[inputs[structure.Z]])
            inputs[structure.position] -= (masses.unsqueeze(-1) * R).sum(
                0
            ) / masses.sum()
            return inputs
        key = (R.device, R.dtype)  # the mass table on the batch's device, once
        if key not in self._masses:
            self._masses[key] = torch.tensor(atomic_masses, dtype=R.dtype).to(R.device)
        masses = self._masses[key][inputs[structure.Z]]
        idx_m = inputs[structure.idx_m]
        center = _segment_mean(R, masses, idx_m, inputs[structure.n_atoms].shape[0])
        inputs[structure.position] = R - center[idx_m]
        return inputs


@register
class SubtractCenterOfGeometry(Transform):
    """
    Subtract center of geometry from positions.

    Works on one structure, or on a collated batch (``idx_m`` present), structure
    by structure.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        R = inputs[structure.position]
        if structure.idx_m not in inputs:
            inputs[structure.position] -= R.mean(0)
            return inputs
        idx_m = inputs[structure.idx_m]
        ones = torch.ones(R.shape[0], dtype=R.dtype, device=R.device)
        center = _segment_mean(R, ones, idx_m, inputs[structure.n_atoms].shape[0])
        inputs[structure.position] = R - center[idx_m]
        return inputs


@register
class NiggliReduce(Transform):
    """
    Replace the cell of a periodic structure by its Niggli cell, the unique
    reduced basis of its lattice, in the cell's own orientation.

    The crystal does not change and the atoms keep their positions; follow
    with :class:`WrapPositions` to move them into the new cell.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cell = inputs[structure.cell]
        inputs[structure.cell] = niggli_reduce(cell.view(3, 3)).view(cell.shape)
        return inputs


@register
class WrapPositions(Transform):
    """
    Wrap atom positions into their periodic cell, along the periodic
    directions only. This routine requires a non-zero cell.

    Works on one structure and on a collated batch.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps (float): small offset for numerical stability; fractional
                coordinates end up in [-eps, 1 - eps).
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cells = atom_cells(inputs)
        frac = to_frac(inputs[structure.R], cells)
        pbc = inputs[structure.pbc].view(-1, 3)[structure_index(inputs)]
        wrapped = (frac + self.eps) % 1.0 - self.eps
        inputs[structure.R] = to_cart(torch.where(pbc, wrapped, frac), cells)
        return inputs


@register
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
            zmax: Size of the atomref table: it holds the atomic numbers below
                zmax. Atomrefs from the data are padded with zeros to this size;
                entries beyond it are an error, as is a structure with an atomic
                number from zmax on.
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

        assert not (estimate_atomref and atomrefs is not None), (
            "You can not set `atomrefs` and use `estimate_atomrefs=True!`"
        )

        if atomrefs is not None:
            self._atomrefs_initialized = True
        else:
            self._atomrefs_initialized = False

        if property_mean is not None:
            self._mean_initialized = True
        else:
            self._mean_initialized = False

        if self.remove_atomrefs:
            atomrefs = atomrefs if atomrefs is not None else torch.zeros((zmax,))
            self.register_buffer("atomref", atomrefs)
        if self.remove_mean:
            if property_mean is None:
                property_mean = torch.zeros((1,))
            self.register_buffer("mean", property_mean)

    def initialize(self, stats) -> None:
        """
        Initialize mean and/or atomrefs from a stats source (any object with
        ``get_stats``/``get_atomrefs``, e.g. the datamodule or its provider).
        """
        if self.remove_atomrefs and not self._atomrefs_initialized:
            atrefs = stats.get_atomrefs(
                self._property, self.is_extensive, estimate=self.estimate_atomref
            )
            self.atomref = _fit_atomrefs(atrefs[self._property], len(self.atomref))

        if self.remove_mean and not self._mean_initialized:
            mean, _std = stats.get_stats(
                self._property, self.is_extensive, self.remove_atomrefs
            )
            # the statistics are 0-d: keep the shape the constructor gave, which a
            # model built from its hyperparameters has
            self.mean = mean.detach().reshape(self.mean.shape)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.remove_mean:
            mean = (
                self.mean * inputs[structure.n_atoms]
                if self.is_extensive
                else self.mean
            )
            inputs[self._property] -= mean

        if self.remove_atomrefs:
            y0i = self.atomref[inputs[structure.Z]]
            if structure.idx_m in inputs:  # a collated batch: per structure
                atomref_bias = scatter_add(
                    y0i,
                    inputs[structure.idx_m],
                    dim_size=inputs[structure.n_atoms].shape[0],
                )
            else:
                atomref_bias = torch.sum(y0i)
            if not self.is_extensive:
                # clamp: a padded batch may hold empty structures (no atoms)
                atomref_bias = atomref_bias / inputs[structure.n_atoms].clamp(min=1)
            inputs[self._property] -= atomref_bias
        return inputs


@register
class ScaleProperty(Transform):
    """
    Scale an entry of the input or results dictionary.

    The `scale` can be obtained automatically from training statistics via
    `initialize(stats)`. Otherwise, it has to be provided in the init
    manually.

    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(
        self,
        input_key: str,
        target_key: str | None = None,
        output_key: str | None = None,
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

        if scale is None:
            scale = torch.ones((1,))
        self.register_buffer("scale", scale)

    def initialize(self, stats) -> None:
        """
        Initialize scaling using training statistics.
        """
        if not self._initialized:
            mean, std = stats.get_stats(self._target_key, True, False)
            scale = mean if self._scale_by_mean else std
            # the statistics are 0-d float64: keep the constructor's shape, which a
            # model built from its hyperparameters has, and its dtype, so that the
            # product keeps the input's dtype (as with the 0-d scale before)
            scale = torch.abs(scale).detach().reshape(self.scale.shape)
            self.scale = scale.to(self.scale.dtype)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        inputs[self.output_key] = inputs[self.input_key] * self.scale
        return inputs


@register
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
            zmax: Size of the atomref table: it holds the atomic numbers below
                zmax. Atomrefs from the data are padded with zeros to this size;
                entries beyond it are an error, as is a structure with an atomic
                number from zmax on.
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

        assert not (estimate_atomref and atomrefs is not None), (
            "You can not set `atomrefs` and use `estimate_atomrefs=True!`"
        )

        if atomrefs is not None:
            self._atomrefs_initialized = True
        else:
            self._atomrefs_initialized = False

        if property_mean is not None:
            self._mean_initialized = True
        else:
            self._mean_initialized = False

        atomrefs = atomrefs if atomrefs is not None else torch.zeros((zmax,))
        if property_mean is None:
            property_mean = torch.zeros((1,))
        self.register_buffer("atomref", atomrefs)
        self.register_buffer("mean", property_mean)

    def initialize(self, stats) -> None:
        """
        Initialize mean and/or atomrefs from a stats source (any object with
        ``get_stats``/``get_atomrefs``, e.g. the datamodule or its provider).
        """
        if self.add_atomrefs and not self._atomrefs_initialized:
            atrefs = stats.get_atomrefs(
                self._property, self.is_extensive, estimate=self.estimate_atomref
            )
            self.atomref = _fit_atomrefs(atrefs[self._property], len(self.atomref))

        if self.add_mean and not self._mean_initialized:
            mean, _std = stats.get_stats(
                self._property, self.is_extensive, self.add_atomrefs
            )
            # 0-d: keep the constructor's shape, as RemoveOffsets does (the dtype
            # stays float64: the offsets are added in place, in the output's dtype)
            self.mean = mean.detach().reshape(self.mean.shape)

    def set_offsets(self, atomrefs: torch.Tensor, mean: torch.Tensor) -> None:
        """Set atomrefs and mean, as if given to the constructor: ``initialize``
        then leaves them alone."""
        atomrefs = _fit_atomrefs(atomrefs, len(self.atomref))
        self.atomref = atomrefs.to(self.atomref.device)
        self.mean = mean.detach().reshape(1).to(self.mean.device)
        self._atomrefs_initialized = True
        self._mean_initialized = True

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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
            maxm = inputs[structure.n_atoms].shape[0]

            y0 = scatter_add(y0i, idx_m, dim_size=maxm)

            if not self.is_extensive:
                # clamp: a padded batch may hold empty structures (no atoms)
                y0 = y0 / inputs[structure.n_atoms].clamp(min=1)

            inputs[self._property] += y0

        return inputs

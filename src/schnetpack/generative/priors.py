"""
Priors: the distribution of the x1 endpoint.

One object serves both sides of a generative model: training draws x1
positions per data sample (:meth:`Prior.sample_positions`), and sampling
starts from the same distribution (:meth:`Prior.sample_from_batch`,
:meth:`Prior.sample`). A prior declares whether its draws are isotropic
Gaussian (:attr:`Prior.gaussian`) and their scale (:attr:`Prior.std`), which
gate the score/noise targets and the SDE chart. A draw may read the batch's
layout and atom types, never the data positions. Design:
``docs_new/priors.md``.
"""

import abc
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn

__all__ = [
    "Prior",
    "GaussianPrior",
    "DatasetPrior",
    "DatasetStructures",
    "StatisticsStructures",
]


class DatasetStructures:
    """
    Structures from a dataset, collated as the dataloader would.

    Successive :meth:`sample` calls walk through the dataset without
    repeating a structure until all were drawn, then start the next pass.
    Note that ``dataset[i]`` applies the dataset's transforms.
    """

    def __init__(self, dataset: Dataset, shuffle: bool = True):
        """
        Args:
            dataset: dataset of structures, e.g. an
                :class:`~schnetpack.data.ASEAtomsData` or a subset of it
            shuffle: walk the dataset in a random order, reshuffled every
                pass; False walks it in order
        """
        if len(dataset) == 0:
            raise ValueError("DatasetStructures needs a non-empty dataset.")
        self.dataset = dataset
        self.shuffle = shuffle
        self._pending: list[int] = []

    def _next_indices(self, n_samples: int) -> list[int]:
        indices: list[int] = []
        while len(indices) < n_samples:
            if not self._pending:
                n = len(self.dataset)
                order = torch.randperm(n) if self.shuffle else torch.arange(n)
                self._pending = order.tolist()
            k = n_samples - len(indices)
            indices += self._pending[:k]
            self._pending = self._pending[k:]
        return indices

    def sample(self, n_samples: int) -> dict[str, torch.Tensor]:
        """Collate the next ``n_samples`` structures of the dataset."""
        return _atoms_collate_fn(
            [self.dataset[i] for i in self._next_indices(n_samples)]
        )


class StatisticsStructures:
    """
    Structures built from composition statistics: atom counts and atom types.

    Each structure draws its atom count from the ``n_atoms`` histogram, then
    each atom its type independently from ``atom_types``; a draw may be a
    composition the dataset does not contain.
    """

    def __init__(self, n_atoms: torch.Tensor, atom_types: torch.Tensor):
        """
        Args:
            n_atoms: histogram of atom counts; entry k weighs structures of k
                atoms. Need not be normalized.
            atom_types: weights of the atom types; entry z weighs atomic
                number z. Need not be normalized.
        """
        n_atoms = torch.as_tensor(n_atoms, dtype=torch.float)
        atom_types = torch.as_tensor(atom_types, dtype=torch.float)
        for name, weights in (("n_atoms", n_atoms), ("atom_types", atom_types)):
            if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError(
                    f"{name} must be a 1-D tensor of non-negative weights "
                    "with a positive sum."
                )
        if n_atoms[0] > 0:
            raise ValueError("n_atoms puts weight on structures of zero atoms.")
        self.n_atoms = n_atoms
        self.atom_types = atom_types

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "StatisticsStructures":
        """Count atom numbers and atom types over all structures of a dataset."""
        zs = [dataset[i][properties.Z] for i in range(len(dataset))]
        return cls(
            n_atoms=torch.bincount(torch.tensor([z.shape[0] for z in zs])),
            atom_types=torch.bincount(torch.cat(zs).long()),
        )

    def sample(self, n_samples: int) -> dict[str, torch.Tensor]:
        """
        Draw ``n_samples`` compositions.

        Returns:
            A batch with ``Z``, ``n_atoms``, ``idx_m`` and ``idx``; the
            positions are the prior's to draw.
        """
        n_atoms = torch.multinomial(self.n_atoms, n_samples, replacement=True)
        z = torch.multinomial(self.atom_types, int(n_atoms.sum()), replacement=True)
        return {
            properties.idx: torch.arange(n_samples),
            properties.Z: z,
            properties.n_atoms: n_atoms,
            properties.idx_m: torch.repeat_interleave(torch.arange(n_samples), n_atoms),
        }


class Prior(abc.ABC):
    """Distribution of the x1 endpoint: training draws and sampling starts."""

    gaussian: bool = False
    """Whether draws are independent isotropic Gaussian of scale :attr:`std`.

    Gates the score/noise parametrizations. Defaults to False: a wrong True
    trains to garbage silently, a wrong False merely raises.
    """

    std: float | None = None
    """Scale of the endpoint, or None when it is not a single number.

    The process's noise level is sigma(t) = b(t) * std; None means every
    consumer that needs sigma raises.
    """

    def __init__(
        self, structures: DatasetStructures | StatisticsStructures | None = None
    ):
        """
        Args:
            structures: source of the structures :meth:`sample` draws; any
                object with ``sample(n_samples) -> batch``. Without one only
                :meth:`sample_from_batch` is available.
        """
        self.structures = structures

    @abc.abstractmethod
    def sample_positions(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """
        Draw x1 positions for a batch: the only draw training makes.

        Args:
            batch: the structures to draw for. Positions, when present, give
                shape, dtype and device (their values are never read);
                otherwise the shape is ``(len(Z), 3)``. The layout
                (``idx_m``) and atom types may be read.

        Returns:
            The positions, shaped like the batch's.
        """
        raise NotImplementedError

    def sample_from_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Copy of ``batch`` with its positions redrawn from the prior."""
        return {**batch, properties.R: self.sample_positions(batch)}

    def sample(self, n_samples: int) -> dict[str, Any]:
        """Draw ``n_samples`` structures from :attr:`structures` with positions from the prior."""
        if self.structures is None:
            raise ValueError(
                f"{type(self).__name__} has no structures to sample from; pass "
                "structures=DatasetStructures(...) or StatisticsStructures(...), "
                "or redraw a given batch with sample_from_batch"
            )
        return self.sample_from_batch(self.structures.sample(n_samples))


def _positions_like(batch: Mapping[str, Any]):
    """Shape, dtype and device of the positions a draw for ``batch`` needs."""
    if properties.R in batch:
        r = batch[properties.R]
        return r.shape, r.dtype, r.device
    if properties.Z in batch:
        z = batch[properties.Z]
        return (z.shape[0], 3), None, z.device
    raise KeyError(
        f"cannot infer the positions' shape: the batch holds neither "
        f"{properties.R!r} nor {properties.Z!r}"
    )


class GaussianPrior(Prior):
    """
    Isotropic zero-mean Gaussian N(0, std^2 I), centered per molecule.

    The endpoint of plain diffusion and flow matching: std = 1 for the
    variance-preserving family, sigma_max for VE. With ``centered=True``
    (default) each segment's mean is subtracted, so x1 lives in the same
    zero-center-of-geometry subspace as data preprocessed with
    :class:`~schnetpack.transform.SubtractCenterOfGeometry`; a
    translation-invariant network cannot learn anything else. Centering
    keeps the draw Gaussian on that subspace, so :attr:`gaussian` stays True,
    but x0 must be centered too. Segments come from ``batch[segment_key]``;
    without it the batch is one group.
    """

    gaussian = True

    def __init__(
        self,
        std: float = 1.0,
        centered: bool = True,
        segment_key: str = properties.idx_m,
        structures: DatasetStructures | StatisticsStructures | None = None,
    ):
        """
        Args:
            std: standard deviation of the endpoint, before centering
            centered: draw in the zero-mean subspace of each segment; set
                False for data without translation symmetry, or when the
                leading axis is independent samples rather than atoms
            segment_key: batch key holding the segment ids (default: the
                molecule index)
            structures: source of the structures :meth:`sample` draws
        """
        super().__init__(structures)
        self.std = std
        self.centered = centered
        self.segment_key = segment_key

    def sample_positions(self, batch):
        shape, dtype, device = _positions_like(batch)
        x = self.std * torch.randn(*shape, dtype=dtype, device=device)
        if not self.centered:
            return x
        return self.center(x, batch.get(self.segment_key))

    @staticmethod
    def center(x: torch.Tensor, segments: torch.Tensor | None) -> torch.Tensor:
        """
        Subtract each segment's mean along the leading axis.

        Args:
            x: batch to center, shape (n, ...)
            segments: segment id per row, shape (n,); None centers x as one
                group
        """
        if segments is None:
            return x - x.mean(0, keepdim=True)
        if segments.shape[0] != x.shape[0]:
            raise ValueError(
                f"segment ids must be one per row: got {segments.shape[0]} "
                f"for {x.shape[0]} rows."
            )
        n_segments = int(segments.max()) + 1 if segments.numel() else 0
        index = segments.reshape(-1, *(1,) * (x.ndim - 1)).expand_as(x)
        totals = torch.zeros(
            n_segments, *x.shape[1:], dtype=x.dtype, device=x.device
        ).scatter_add_(0, index, x)
        counts = torch.zeros(n_segments, dtype=x.dtype, device=x.device).scatter_add_(
            0, segments, torch.ones_like(segments, dtype=x.dtype)
        )
        means = totals / counts.reshape(-1, *(1,) * (x.ndim - 1))
        return x - means[segments]


class DatasetPrior(Prior):
    """
    Stored structures returned unchanged, positions included: the start of a
    relaxation (e.g. :class:`~schnetpack.dynamics.relax.DirectDenoising`).

    Has no positions law, so it cannot serve as a training endpoint.
    """

    def __init__(self, dataset: Dataset, shuffle: bool = True):
        """
        Args:
            dataset: dataset of the structures to start from
            shuffle: draw them in a random order; False draws them in order
        """
        super().__init__(DatasetStructures(dataset, shuffle=shuffle))

    def sample_positions(self, batch):
        raise TypeError(
            "DatasetPrior has no positions law: it returns stored structures "
            "and serves as a sampling start only, not as a training endpoint."
        )

    def sample_from_batch(self, batch):
        return dict(batch)

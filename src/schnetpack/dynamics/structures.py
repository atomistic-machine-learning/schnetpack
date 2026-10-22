"""
Structure providers: where the structures a driver starts from come from.

A provider draws ``n_samples`` structures as a collated batch: atom types and
layout, and whatever else its source holds (positions, cells). A
:class:`~schnetpack.generative.priors.Prior` then redraws the keys it owns;
without one the structures are run as given, e.g. to relax stored geometries.
"""

import abc

import torch
from torch.utils.data import Dataset

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn

__all__ = ["StructureProvider", "DatasetStructures", "StatisticsStructures"]


class StructureProvider(abc.ABC):
    """Source of starting structures: ``n_samples -> batch``."""

    @abc.abstractmethod
    def __call__(self, n_samples: int) -> dict[str, torch.Tensor]:
        """
        Draw ``n_samples`` structures.

        Returns:
            A collated batch with at least ``n_atoms``, ``idx_m`` and ``idx``,
            and ``Z`` unless the types are generated.
        """
        raise NotImplementedError


class DatasetStructures(StructureProvider):
    """
    Structures from a dataset, collated as the dataloader would.

    Successive calls walk through the dataset without repeating a structure
    until all were drawn, then start the next pass. Note that ``dataset[i]``
    applies the dataset's transforms.
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

    def __call__(self, n_samples):
        """Collate the next ``n_samples`` structures of the dataset."""
        return _atoms_collate_fn(
            [self.dataset[i] for i in self._next_indices(n_samples)]
        )


class StatisticsStructures(StructureProvider):
    """
    Structures built from composition statistics: atom counts and atom types.

    Each structure draws its atom count from the ``n_atoms`` histogram, then
    each atom its type independently from ``atom_types``; a draw may be a
    composition the dataset does not contain. Without ``atom_types`` the
    batch has no types, for a variable that generates them (its prior writes
    them). The batch has no positions and a zero cell. Structures are
    non-periodic, as in a collated molecule dataset, or periodic with
    ``pbc``, for a cell variable that writes the cell.
    """

    def __init__(
        self,
        n_atoms: torch.Tensor,
        atom_types: torch.Tensor | None,
        pbc: bool = False,
    ):
        """
        Args:
            n_atoms: histogram of atom counts; entry k weighs structures of k
                atoms. Need not be normalized.
            atom_types: weights of the atom types; entry z weighs atomic
                number z. Need not be normalized. None draws no types.
            pbc: periodic in all three directions
        """
        n_atoms = torch.as_tensor(n_atoms, dtype=torch.float)
        histograms = {"n_atoms": n_atoms}
        if atom_types is not None:
            atom_types = torch.as_tensor(atom_types, dtype=torch.float)
            histograms["atom_types"] = atom_types
        for name, weights in histograms.items():
            if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError(
                    f"{name} must be a 1-D tensor of non-negative weights "
                    "with a positive sum."
                )
        if n_atoms[0] > 0:
            raise ValueError("n_atoms puts weight on structures of zero atoms.")
        self.n_atoms = n_atoms
        self.atom_types = atom_types
        self.pbc = pbc

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, atom_types: bool = True, pbc: bool = False
    ) -> "StatisticsStructures":
        """
        Count atom numbers, and atom types if asked, over all structures of a
        dataset.

        Args:
            dataset: the structures to count
            atom_types: False to draw atom counts only, for generated types
            pbc: draw periodic structures
        """
        zs = [dataset[i][properties.Z] for i in range(len(dataset))]
        return cls(
            n_atoms=torch.bincount(torch.tensor([z.shape[0] for z in zs])),
            atom_types=torch.bincount(torch.cat(zs).long()) if atom_types else None,
            pbc=pbc,
        )

    def __call__(self, n_samples):
        """
        Draw ``n_samples`` compositions.

        Returns:
            A batch with ``n_atoms``, ``idx_m``, ``idx``, ``cell``, ``pbc``
            and, with ``atom_types``, ``Z``; the positions are the prior's to
            draw.
        """
        n_atoms = torch.multinomial(self.n_atoms, n_samples, replacement=True)
        batch = {
            properties.idx: torch.arange(n_samples),
            properties.n_atoms: n_atoms,
            properties.idx_m: torch.repeat_interleave(torch.arange(n_samples), n_atoms),
            properties.cell: torch.zeros(n_samples, 3, 3),
            properties.pbc: torch.full((3 * n_samples,), self.pbc, dtype=torch.bool),
        }
        if self.atom_types is not None:
            batch[properties.Z] = torch.multinomial(
                self.atom_types, int(n_atoms.sum()), replacement=True
            )
        return batch

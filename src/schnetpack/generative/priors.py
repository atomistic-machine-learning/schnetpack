"""
Priors — the distribution of the x1 endpoint.

A prior answers one question — what x1 *is* — and it is asked twice:

- **Training.** The forward process needs an endpoint for every data sample:
  :meth:`Prior.sample_positions` draws x1 positions for the batch being
  diffused, and the coupling then decides how the two are paired. Under the
  default :class:`~schnetpack.generative.couplings.IdentityCoupling` this is
  the familiar "fresh noise per sample". Training only ever draws positions.
- **Sampling.** With b(t_max) = 1 the state the reverse process starts from
  *is* the x1 endpoint, so the correct start distribution is x1's marginal.
  When the coupling only re-pairs (any marginal-preserving coupling), that
  marginal is this prior itself, and
  :meth:`~schnetpack.generative.processes.Process.sampling_prior`
  hands the very same object to the
  :class:`~schnetpack.dynamics.sampling.sampler.Sampler`.

  Sampling starts from a full batch, as the dataloader would give it:
  :meth:`Prior.sample_from_batch` redraws the positions of a given batch (a
  test-set batch, say), and :meth:`Prior.sample` draws ``n_samples``
  structures from :attr:`Prior.structures` — a dataset
  (:class:`DatasetStructures`) or composition statistics
  (:class:`StatisticsStructures`) — and redraws their positions.

One object serving both sides is the point: train-time and sample-time x1
cannot drift apart, because there is nothing to restate.

A prior also declares what its draws *are*:

- :attr:`Prior.gaussian` says whether they are independent isotropic
  Gaussian. It gates the score/noise parametrizations and the process's
  Gaussian-only closed forms — via
  :meth:`~schnetpack.generative.processes.Process.gaussian_kernel_obstruction`
  — and defaults to False, because a wrong True trains to garbage silently
  while a wrong False merely raises.
- :attr:`Prior.std` is the endpoint's scale, or None when it has no single
  scalar value (a shape prior with per-molecule covariance). The process's
  noise level is sigma(t) = b(t) * std, so everything that needs sigma —
  score conversions, the SDE diffusion, churn > 0 sampling — needs a
  declared std. For a VE process std is sigma_max, the knob that must match
  the data scale (see :class:`~schnetpack.generative.processes.VE`).

The batch a draw receives is what an endpoint may legitimately depend on —
atom types, atom count, the layout (``idx_m``), scaffold indices — never the
data positions themselves. A distribution shaped by the *data values* is a
coupling, not a prior (see
:class:`~schnetpack.generative.couplings.PCVarianceCoupling`).

:class:`DatasetPrior` is the one prior without a positions law: it returns
stored structures (non-equilibrium ones, to relax) unchanged, so it serves
as a sampling start only.
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
    repeating a structure until all of them were drawn, then start the next
    pass — like the epochs of a dataloader.
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
    each atom its type independently from the ``atom_types`` probabilities.
    The compositions follow the dataset's statistics, not its molecules — a
    draw may be a composition the dataset does not contain.
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
    """Distribution of the x1 endpoint — training draws and sampling starts."""

    gaussian: bool = False
    """Whether draws are independent isotropic Gaussian of scale :attr:`std`.

    Gates the score/noise parametrizations, whose training targets *are*
    statements about a Gaussian endpoint. A custom prior must opt in
    explicitly.
    """

    std: float | None = None
    """Scale of the endpoint, or None when it is not a single number.

    The process reads its noise level sigma(t) = b(t) * std from this; None
    means conversions and samplers that need sigma raise and ask for it
    explicitly.
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
        Draw x1 positions for a batch — the tensor-level law, and the only
        draw training makes.

        Args:
            batch: the structures to draw for. Positions, when present, give
                shape, dtype and device (their values are never read);
                otherwise the shape is ``(len(Z), 3)``. The layout
                (``idx_m``) and atom types are there to be read.

        Returns:
            The positions, shaped like the batch's.
        """
        raise NotImplementedError

    def sample_from_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Copy of ``batch`` with its positions redrawn from the prior."""
        return {**batch, properties.R: self.sample_positions(batch)}

    def sample(self, n_samples: int) -> dict[str, Any]:
        """
        Draw ``n_samples`` starting structures from :attr:`structures`, with
        positions drawn from the prior.
        """
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

    The endpoint of every plain diffusion and flow-matching process: std = 1
    for the variance-preserving family, sigma_max for VE — where it must
    match the data scale (largest pairwise distance rule; see
    :class:`~schnetpack.generative.processes.VE`). Exact as a sampling start
    when a(t_max) = 0 (flow matching); for the diffusion paths it is the
    usual approximation that the residual a(t_max) x0 term is negligible.

    :attr:`centered` (default True) subtracts each molecule's mean from its
    draw, putting x1 in the same zero-COM subspace that
    :class:`~schnetpack.transform.SubtractCenterOfGeometry` puts x0 in. That
    is what molecules need: a translation-invariant network can never predict
    a displacement of a whole structure, so an off-subspace endpoint is
    unlearnable noise in every training target and an offset nothing removes
    in every sampling start. Uncentered, a draw carries a center of geometry
    of scale ``std * sqrt(d / n)`` per molecule (n atoms in d dimensions) —
    4.6 A for a 12-atom molecule at ``std = 10``, larger than the molecule.

    **Centering does not cost the Gaussian kernel.** Projecting a standard
    normal onto a subspace gives a standard normal *on that subspace*, with
    the same per-direction variance, so :attr:`gaussian` stays True and the
    score/noise parametrizations stay exact. Only the space changes — which
    makes the precondition load-bearing: **x0 must be centered too** (compose
    ``SubtractCenterOfGeometry`` before
    :class:`~schnetpack.generative.transforms.Diffuse`). Centered noise on
    uncentered data leaves x_t's mean drifting with a(t), and the kernel is no
    longer the one the targets assume.

    Which rows share a mean is read from the batch's ``segment_key``
    (``idx_m``), so centering is per molecule rather than per batch. A batch
    without it is one group — correct for a transform running per structure
    inside the dataloader, where the batch *is* one molecule.

    Set ``centered=False`` for data with no translation symmetry to quotient
    out, or when the leading axis is independent samples rather than the atoms
    of one structure — centering couples the rows it spans.
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
            centered: draw in the zero-mean subspace of each segment
            segment_key: batch key holding the segment ids; defaults to
                SchNetPack's molecule index
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
    Stored structures, returned unchanged — the start of a relaxation.

    Draws the dataset's own structures (non-equilibrium ones, to relax) as
    they are, positions included. It has no positions law, so it cannot
    serve as a training endpoint; it is a sampling start only, e.g. for
    :class:`~schnetpack.dynamics.relax.DirectDenoising`.
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

"""
Priors: the distribution of the x1 endpoint.

One object serves both sides of a generative model: training draws x1 for
each data batch, and sampling starts from the same distribution. A prior
maps a batch to a copy with its :attr:`Prior.keys` redrawn; the structures
themselves (atom counts, types) come from a
:class:`~schnetpack.dynamics.structures.StructureProvider`. A prior declares
whether its draws are isotropic Gaussian (:attr:`Prior.gaussian`) and their
scale (:attr:`Prior.std`), which gate the score/noise targets and the SDE
chart. A draw may read the batch's layout and atom types, never the data
values of the keys it draws.
"""

import abc
from collections.abc import Mapping
from typing import Any

import torch

from schnetpack import properties

__all__ = ["Prior", "GaussianPrior"]


class Prior(abc.ABC):
    """Distribution of the x1 endpoint: training draws and sampling starts."""

    keys: tuple[str, ...] = (properties.R,)
    """Batch keys a draw overwrites."""

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

    @abc.abstractmethod
    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """
        Draw x1 for a batch.

        Args:
            batch: the structures to draw for. Values under :attr:`keys`,
                when present, give shape, dtype and device but are never
                read; the layout (``idx_m``) and atom types may be read.

        Returns:
            A copy of ``batch`` with :attr:`keys` redrawn.
        """
        raise NotImplementedError


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
    ):
        """
        Args:
            std: standard deviation of the endpoint, before centering
            centered: draw in the zero-mean subspace of each segment; set
                False for data without translation symmetry, or when the
                leading axis is independent samples rather than atoms
            segment_key: batch key holding the segment ids (default: the
                molecule index)
        """
        self.std = std
        self.centered = centered
        self.segment_key = segment_key

    def __call__(self, batch):
        shape, dtype, device = _positions_like(batch)
        x = self.std * torch.randn(*shape, dtype=dtype, device=device)
        if self.centered:
            x = self.center(x, batch.get(self.segment_key))
        return {**batch, properties.R: x}

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

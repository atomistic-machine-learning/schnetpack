"""
Priors — the distribution of the x1 endpoint.

A prior answers one question — what x1 *is* — and it is asked twice:

- **Training.** The forward process needs an endpoint for every data sample:
  :meth:`Prior.sample_like` draws x1 shaped like x0, and the coupling then
  decides how the two batches are paired. Under the default
  :class:`~schnetpack.generative.couplings.IdentityCoupling` this is the
  familiar "fresh noise per sample".
- **Sampling.** With b(t_max) = 1 the state the reverse process starts from
  *is* the x1 endpoint, so the correct start distribution is x1's marginal.
  When the coupling only re-pairs (any marginal-preserving coupling), that
  marginal is this prior itself, and
  :meth:`~schnetpack.generative.processes.Process.sampling_prior`
  hands the very same object to the
  :class:`~schnetpack.generative.sampler.Sampler`.

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

The ``context`` argument is what an endpoint may legitimately depend on at
generation time too — composition, atom count, scaffold indices, the batch
layout a draw must respect — never the data values themselves. A distribution
shaped by the *data batch* is a coupling, not a prior (see
:class:`~schnetpack.generative.couplings.PCVarianceCoupling`). Structure is
not values: :class:`GaussianPrior` reads ``idx_m`` out of the context to
center each molecule's draw on its own, which is available at generation time
and says nothing about where the atoms go.

Structured priors (per-molecule covariance, scaffolds, second datasets) plug
in through this same interface; starting below t_max from a structured state
pairs with :meth:`~schnetpack.generative.sampler.Sampler.denoise`.
"""

import abc
from typing import Optional, Sequence

import torch

from schnetpack import properties

__all__ = ["Prior", "GaussianPrior"]


class Prior(abc.ABC):
    """Distribution of the x1 endpoint — training draws and sampling starts."""

    gaussian: bool = False
    """Whether draws are independent isotropic Gaussian of scale :attr:`std`.

    Gates the score/noise parametrizations, whose training targets *are*
    statements about a Gaussian endpoint. A custom prior must opt in
    explicitly.
    """

    std: Optional[float] = None
    """Scale of the endpoint, or None when it is not a single number.

    The process reads its noise level sigma(t) = b(t) * std from this; None
    means conversions and samplers that need sigma raise and ask for it
    explicitly.
    """

    @abc.abstractmethod
    def sample(
        self,
        shape: Sequence[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        context=None,
    ) -> torch.Tensor:
        """
        Draw endpoints of the given shape — the generation-side entry.

        Args:
            shape: shape of the batch, (n_samples, ...)
            dtype: dtype of the draw
            device: device of the draw
            context: generation-time conditioning (composition, atom count,
                scaffold indices); ignored by unconditional priors
        """
        raise NotImplementedError

    def sample_like(self, x0: torch.Tensor, context=None) -> torch.Tensor:
        """
        Draw endpoints shaped like a data batch — the training-side entry.

        Defaults to :meth:`sample` with x0's shape, dtype and device, so a
        prior only ever defines its law once. Override only when the training
        draw needs more than the shape (e.g. a scaffold prior copying fixed
        atoms out of the context).
        """
        return self.sample(x0.shape, dtype=x0.dtype, device=x0.device, context=context)


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

    Which rows share a mean comes from ``context``, so centering is per
    molecule rather than per batch:

    - a mapping (a SchNetPack batch dict): ``segment_key`` is read out of it.
      :class:`~schnetpack.generative.transforms.Diffuse` passes the batch it
      is diffusing, and
      :meth:`~schnetpack.generative.sampler.Sampler.sample` forwards whatever
      it is given, so both sides supply ``idx_m`` on their own.
    - a 1-D integer tensor: segment ids directly, one per row.
    - ``None``, or a mapping without ``segment_key``: the whole leading axis
      is one group. Correct where that axis *is* one molecule — a transform
      running per structure inside the dataloader — and wrong for a collated
      batch, which is why the atomistic paths pass their layout rather than
      relying on this.

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
    ):
        """
        Args:
            std: standard deviation of the endpoint, before centering
            centered: draw in the zero-mean subspace of each segment
            segment_key: key holding the segment ids when ``context`` is a
                mapping; defaults to SchNetPack's molecule index
        """
        self.std = std
        self.centered = centered
        self.segment_key = segment_key

    def sample(self, shape, dtype=None, device=None, context=None):
        x = self.std * torch.randn(*shape, dtype=dtype, device=device)
        if not self.centered:
            return x
        return self.center(x, self.segments(context))

    def segments(self, context) -> Optional[torch.Tensor]:
        """Resolve ``context`` to per-row segment ids, or None for one group."""
        if context is None or torch.is_tensor(context):
            return context
        if hasattr(context, "get"):
            return context.get(self.segment_key)
        raise TypeError(
            f"{type(self).__name__} takes context as segment ids, a mapping "
            f"holding {self.segment_key!r}, or None for a single group; got "
            f"{type(context).__name__}."
        )

    @staticmethod
    def center(x: torch.Tensor, segments: Optional[torch.Tensor]) -> torch.Tensor:
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

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
generation time too — composition, atom count, scaffold indices — never the
data values themselves. A distribution shaped by the *data batch* is a
coupling, not a prior (see
:class:`~schnetpack.generative.couplings.PCVarianceCoupling`).

Structured priors (per-molecule covariance, scaffolds, second datasets) plug
in through this same interface; starting below t_max from a structured state
pairs with :meth:`~schnetpack.generative.sampler.Sampler.denoise`.
"""

import abc
from typing import Optional, Sequence

import torch

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
    Isotropic zero-mean Gaussian N(0, std^2 I).

    The endpoint of every plain diffusion and flow-matching process: std = 1
    for the variance-preserving family, sigma_max for VE — where it must
    match the data scale (largest pairwise distance rule; see
    :class:`~schnetpack.generative.processes.VE`). Exact as a sampling start
    when a(t_max) = 0 (flow matching); for the diffusion paths it is the
    usual approximation that the residual a(t_max) x0 term is negligible.
    """

    gaussian = True

    def __init__(self, std: float = 1.0):
        """
        Args:
            std: standard deviation of the endpoint
        """
        self.std = std

    def sample(self, shape, dtype=None, device=None, context=None):
        return self.std * torch.randn(*shape, dtype=dtype, device=device)

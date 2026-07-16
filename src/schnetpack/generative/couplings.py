"""
Couplings — how (x0, x1) endpoint pairs are drawn.

The path fixes the marginals; the coupling fixes the joint law the marginals
leave open. Diffusion and vanilla flow matching pair every data point with
fresh independent noise, which is why the axis usually goes unnoticed. It
becomes the whole story for minibatch-OT flow matching, bridge matching and
Schrödinger bridges, where x1 is a *matched* endpoint rather than a free draw.

Only the training loss (:mod:`schnetpack.generative.losses`) consumes this;
sampling never does, since by then the endpoints are whatever the prior gives.
"""

import abc
from typing import Optional, Tuple

import torch

__all__ = [
    "Coupling",
    "IndependentCoupling",
    "OTCoupling",
    "DataToDataCoupling",
]


class Coupling(abc.ABC):
    """Joint law over endpoint pairs (x0, x1)."""

    @abc.abstractmethod
    def sample(
        self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pair up a batch of data with prior endpoints.

        Args:
            x0: data batch, shape (n_samples, ...)
            x1: candidate prior endpoints; drawn or ignored depending on the
                coupling

        Returns:
            (x0, x1), both shaped like the input x0.
        """
        raise NotImplementedError


class IndependentCoupling(Coupling):
    """
    Product coupling pi = p0 x p1: every sample gets fresh standard normal noise.

    This is what VE, VP, EDM and plain flow matching use — and the reason their
    score and noise training targets are valid, since x1 then *is* the noise
    realization.
    """

    def sample(self, x0, x1=None):
        """Draw fresh noise; a given ``x1`` is ignored by construction."""
        return x0, torch.randn_like(x0)


class OTCoupling(Coupling):
    """
    Minibatch optimal-transport pairing (OT flow matching / rectified flow).

    Solves the OT problem between the data and noise batches and permutes x1 to
    match x0, which straightens the learned velocity field and cuts the number
    of sampling steps. Lands with the OT milestone; the plan is a POT-based
    ``emd`` solve over the squared-distance cost, falling back to a torch-only
    Sinkhorn when POT is absent.
    """

    def sample(self, x0, x1=None):
        raise NotImplementedError(
            "OTCoupling lands with the optimal-transport milestone. "
            "Use IndependentCoupling for standard flow matching."
        )


class DataToDataCoupling(Coupling):
    """
    Paired endpoints for bridge matching: x1 is a data point, not noise.

    Both endpoints come from data (or from a previous bridge iterate), which is
    what turns matching into a Schrödinger-bridge half-step. Requires a path
    with a nonzero :meth:`~schnetpack.generative.paths.Path.gamma`, since a
    bridge's x_t carries its own noise on top of the two endpoints. Note that
    the score and noise targets are invalid here — regress the velocity.
    """

    def sample(self, x0, x1=None):
        raise NotImplementedError(
            "DataToDataCoupling lands with the Schrödinger-bridge milestone."
        )

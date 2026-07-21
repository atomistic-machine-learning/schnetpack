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
    "PermutationCoupling",
    "PCVarianceCoupling",
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


class PermutationCoupling(Coupling):
    """
    Reorder the prior endpoints by their optimal assignment against the data.

    A single-sample special case of :class:`OTCoupling`: the transport plan
    between the two point sets is constrained to a *permutation*, so every x0
    row keeps exactly one x1 partner. Solving the linear assignment problem
    under a squared-distance cost gives the permutation of x1 that minimizes the
    total straight-line transport, which shortens and de-crosses the paths the
    model has to learn.

    The leading axis is treated as the sample axis of one point cloud and the
    rest is flattened into the cost's feature vector — for positions that is
    atoms in 3D, so the assignment pairs each atom with the nearest noise point.
    Because a Gaussian prior is exchangeable, permuting x1 leaves its marginal
    untouched; only the joint with x0 changes. A provided ``x1`` is reordered in
    place of fresh noise, which lets a caller inject constrained noise (e.g.
    zero-COM) and still get the assignment.

    Needs SciPy for the exact solve (``scipy.optimize.linear_sum_assignment``).
    """

    def __init__(self, cost_power: float = 2.0):
        """
        Args:
            cost_power: exponent on the pairwise Euclidean distance used as the
                assignment cost. 2.0 is the squared-distance (OT) cost; 1.0 is
                plain distance.
        """
        self.cost_power = cost_power

    def sample(self, x0, x1=None):
        if x1 is None:
            x1 = torch.randn_like(x0)

        a = x0.reshape(x0.shape[0], -1)
        b = x1.reshape(x1.shape[0], -1)
        cost = torch.cdist(a, b) ** self.cost_power

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as err:
            raise ImportError(
                "PermutationCoupling needs SciPy for the assignment solve; "
                "install scipy or use IndependentCoupling."
            ) from err

        _, col = linear_sum_assignment(cost.detach().cpu().numpy())
        perm = torch.as_tensor(col, device=x1.device, dtype=torch.long)
        return x0, x1[perm]


class PCVarianceCoupling(Coupling):
    """
    Reshape the prior's variance ellipsoid to match the data's.

    Diffusion's isotropic Gaussian prior is a round cloud; a molecule is not.
    This coupling computes the principal axes of x1, then rescales each
    principal component so the variance of x1 along its k-th axis equals the
    variance of x0 along x0's k-th axis (both sorted descending). The prior
    keeps its own random orientation but takes on the data's *shape* — a long
    molecule is met by an elongated noise cloud — so the transport is closer to
    a rotation than a stretch.

    Unlike :class:`PermutationCoupling` this changes x1's marginal law, so the
    noise/score training targets no longer hold; pair it with a velocity
    parametrization. The leading axis is the sample axis and the rest is
    flattened, so for positions the principal axes are the point cloud's 3D
    geometric axes. A provided ``x1`` is reshaped in place of fresh noise.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: floor added to both variances before the ratio, guarding the
                rescale against a degenerate (near-zero-variance) principal
                axis.
        """
        self.eps = eps

    def sample(self, x0, x1=None):
        if x1 is None:
            x1 = torch.randn_like(x0)

        a = x0.reshape(x0.shape[0], -1)
        b = x1.reshape(x1.shape[0], -1)
        n = a.shape[0]

        b_mean = b.mean(0, keepdim=True)
        a_c = a - a.mean(0, keepdim=True)
        b_c = b - b_mean

        # principal axes (rows of V) and per-axis variance, both sorted desc.
        var_a = torch.linalg.svdvals(a_c) ** 2 / max(n - 1, 1)
        _, s_b, v_b = torch.linalg.svd(b_c, full_matrices=False)
        var_b = s_b**2 / max(n - 1, 1)

        k = v_b.shape[0]
        scale = torch.sqrt((var_a[:k] + self.eps) / (var_b + self.eps))

        proj = (b_c @ v_b.T) * scale.unsqueeze(0)  # coords in x1's PC frame
        b_new = proj @ v_b + b_mean
        return x0, b_new.reshape_as(x1)


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

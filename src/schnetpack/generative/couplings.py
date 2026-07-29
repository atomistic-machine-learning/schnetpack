"""
Couplings — how already-drawn endpoint batches are paired.

The prior decides what x1 *is*; the path decides *when* it takes over. The
coupling sits between them and fixes the joint law of (x0, x1): given a data
batch and a batch of prior draws, which x1 goes with which x0. The default
leaves the pairing alone — diffusion and vanilla flow matching pair every
sample with the fresh draw it was handed, which is why the axis usually goes
unnoticed. It becomes the whole story for minibatch-OT flow matching and for
alignment tricks that shorten and de-cross the transport paths a model has
to learn.

A coupling never draws x1 — that is the prior's job (see
:mod:`schnetpack.generative.priors`). What it must declare is what its
pairing does to x1's *marginal*:

- :attr:`Coupling.preserves_marginal` is True for anything that at most
  re-orders x1 across the batch (identity, permutation, OT): re-pairing
  leaves the marginal law untouched, which is the property that lets the
  training prior double as the sampling start
  (:meth:`~schnetpack.generative.processes.Process.sampling_prior`),
  and — for exchangeable Gaussian draws — keeps the score/noise targets
  valid (judged by
  :meth:`~schnetpack.generative.processes.Process.gaussian_kernel_obstruction`).
  It is False
  for anything that reshapes x1 from the data's values, where no data-free
  start distribution exists and the sampler demands an explicit
  :class:`~schnetpack.generative.priors.Prior`. The default is False: a
  wrong True starts sampling from the wrong distribution silently, a wrong
  False merely asks for an explicit prior.
"""

import abc
from typing import Tuple

import torch

__all__ = [
    "Coupling",
    "IdentityCoupling",
    "PermutationCoupling",
    "PCVarianceCoupling",
    "OTCoupling",
]


class Coupling(abc.ABC):
    """Joint law of the endpoint pair (x0, x1), as a re-pairing of batches."""

    preserves_marginal: bool = False
    """Whether :meth:`pair` leaves x1's marginal law untouched.

    True for pure re-orderings; False for anything data-dependent. Defaults
    to False — a custom coupling must opt in explicitly, because a wrong
    True samples from the wrong start silently while a wrong False merely
    demands an explicit prior.
    """

    @abc.abstractmethod
    def pair(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-pair a data batch with a batch of prior draws.

        Args:
            x0: data batch, shape (n_samples, ...)
            x1: prior endpoints, shaped like x0

        Returns:
            (x0, x1), both shaped like the inputs.
        """
        raise NotImplementedError


class IdentityCoupling(Coupling):
    """
    Leave the pairing exactly as drawn: the product coupling pi = p0 x p1.

    Every sample keeps the fresh endpoint the prior handed it — what VE, VP
    and plain flow matching use, and the reason their score and noise
    training targets are valid: x1 stays the independent noise realization
    the kernel math assumes.
    """

    preserves_marginal = True

    def pair(self, x0, x1):
        return x0, x1


class PermutationCoupling(Coupling):
    """
    Reorder the prior endpoints by their optimal assignment against the data.

    A single-batch special case of :class:`OTCoupling`: the transport plan
    between the two point sets is constrained to a *permutation*, so every x0
    row keeps exactly one x1 partner. Solving the linear assignment problem
    under a squared-distance cost gives the permutation of x1 that minimizes
    the total straight-line transport, which shortens and de-crosses the
    paths the model has to learn.

    The leading axis is treated as the sample axis of one point cloud and the
    rest is flattened into the cost's feature vector — for positions that is
    atoms in 3D, so the assignment pairs each atom with the nearest noise
    point. Because re-ordering exchangeable draws leaves the marginal
    untouched, ``preserves_marginal`` holds; only the joint with x0 changes.

    Needs SciPy for the exact solve (``scipy.optimize.linear_sum_assignment``).
    """

    preserves_marginal = True

    def __init__(self, cost_power: float = 2.0):
        """
        Args:
            cost_power: exponent on the pairwise Euclidean distance used as
                the assignment cost. 2.0 is the squared-distance (OT) cost;
                1.0 is plain distance.
        """
        self.cost_power = cost_power

    def pair(self, x0, x1):
        a = x0.reshape(x0.shape[0], -1)
        b = x1.reshape(x1.shape[0], -1)
        cost = torch.cdist(a, b) ** self.cost_power

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as err:
            raise ImportError(
                "PermutationCoupling needs SciPy for the assignment solve; "
                "install scipy or use IdentityCoupling."
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
    molecule is met by an elongated noise cloud — so the transport is closer
    to a rotation than a stretch.

    Unlike :class:`PermutationCoupling` this changes x1's marginal law from
    the data's values (``preserves_marginal`` is False), with two enforced
    consequences: the process loses its Gaussian kernel (use a velocity, x0
    or pseudo-force parametrization — the score/noise ones refuse), and
    sampling needs an explicit
    :class:`~schnetpack.generative.priors.Prior` whose covariance matches the
    statistics trained under. The leading axis is the sample axis and the
    rest is flattened, so for positions the principal axes are the point
    cloud's 3D geometric axes.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: floor added to both variances before the ratio, guarding the
                rescale against a degenerate (near-zero-variance) principal
                axis.
        """
        self.eps = eps

    def pair(self, x0, x1):
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

    Solves the OT problem between the data and prior batches and permutes x1
    to match x0, which straightens the learned velocity field and cuts the
    number of sampling steps. Lands with the OT milestone; the plan is a
    POT-based ``emd`` solve over the squared-distance cost, falling back to a
    torch-only Sinkhorn when POT is absent. Like the permutation special
    case, re-pairing leaves x1's marginal untouched.
    """

    preserves_marginal = True

    def pair(self, x0, x1):
        raise NotImplementedError(
            "OTCoupling lands with the optimal-transport milestone. "
            "Use IdentityCoupling for standard flow matching."
        )

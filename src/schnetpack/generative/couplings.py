"""
Couplings: how already-drawn endpoint batches are paired.

A coupling fixes the joint law of (x0, x1) by re-pairing a data batch with a
batch of prior draws; it never draws x1 itself. Re-pairing can be restricted
to interchangeable rows via ``groups`` (for molecules: ``(idx_m, Z)``). A
coupling declares whether it preserves x1's marginal
(:attr:`Coupling.preserves_marginal`, needed to reuse the training prior as
the sampling start) and whether it pairs independently of the values
(:attr:`Coupling.independent_pairs`, needed for the Gaussian kernel).
Design and catalog: ``docs_new/couplings.md``.
"""

import abc

import torch

__all__ = [
    "Coupling",
    "IdentityCoupling",
    "PermutationCoupling",
    "PCVarianceCoupling",
    "OTCoupling",
]


def row_blocks(groups: torch.Tensor | None, n: int, device=None) -> list[torch.Tensor]:
    """
    Row indices grouped into blocks whose members may exchange endpoints.

    Args:
        groups: one label per row, shape (n,), or several label columns,
            shape (n, k), in which case rows must agree on every column to
            share a block. None puts every row in one block.
        n: number of rows the labels must cover
        device: device for the returned index tensors

    Returns:
        A list of 1-D index tensors partitioning ``range(n)``.
    """
    if groups is None:
        return [torch.arange(n, device=device)]
    labels = torch.as_tensor(groups)
    if labels.ndim == 1:
        labels = labels.unsqueeze(-1)
    if labels.shape[0] != n:
        raise ValueError(
            f"groups covers {labels.shape[0]} rows but the batch has {n}; "
            "label every row of the sample axis (for positions: every atom)"
        )
    blocks = []
    for key in torch.unique(labels, dim=0):
        rows = (labels == key).all(-1).nonzero(as_tuple=False).squeeze(-1)
        blocks.append(rows.to(device))
    return blocks


class Coupling(abc.ABC):
    """Joint law of the endpoint pair (x0, x1), as a re-pairing of batches."""

    preserves_marginal: bool = False
    """Whether :meth:`pair` leaves x1's marginal law untouched.

    True for pure re-orderings, False for anything that reshapes the values.
    Defaults to False: a wrong True samples from the wrong start silently.
    """

    independent_pairs: bool = False
    """Whether :meth:`pair` assigns endpoints without looking at the values.

    Implies :attr:`preserves_marginal`, but not conversely: an optimal
    assignment keeps the marginal while biasing each x0's partner toward it.
    Defaults to False: a wrong True trains a biased score/noise head silently.
    """

    @abc.abstractmethod
    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        groups: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Re-pair a data batch with a batch of prior draws.

        Args:
            x0: data batch, shape (n_samples, ...)
            x1: prior endpoints, shaped like x0
            groups: labels restricting which rows may exchange endpoints,
                shape (n_samples,) or (n_samples, k); see :func:`row_blocks`.
                None leaves the assignment unrestricted.

        Returns:
            (x0, x1), both shaped like the inputs.
        """
        raise NotImplementedError


class IdentityCoupling(Coupling):
    """
    Leave the pairing as drawn: the product coupling used by VE, VP and plain
    flow matching.
    """

    preserves_marginal = True
    independent_pairs = True

    def pair(self, x0, x1, groups=None):
        # the identity permutation is block-diagonal under any labelling, so
        # `groups` is satisfied by construction
        return x0, x1


class PermutationCoupling(Coupling):
    """
    Reorder the prior endpoints by their optimal assignment against the data.

    Solves the linear assignment problem under a distance cost, per block of
    ``groups``, so each x0 row keeps exactly one x1 partner; the leading axis
    is the sample axis and the rest is flattened. Preserves the marginal but
    looks at the values, so the Gaussian kernel is lost: train a velocity,
    x0 or pseudo-force head on it. Needs SciPy.
    """

    preserves_marginal = True

    def __init__(self, cost_power: float = 2.0):
        """
        Args:
            cost_power: exponent on the pairwise Euclidean distance used as
                the assignment cost; 2.0 is the squared-distance (OT) cost
        """
        self.cost_power = cost_power

    def pair(self, x0, x1, groups=None):
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

        # one solve per block. Restricting the assignment this way is cheaper
        # than a global solve, not just narrower: sum of cubes beats the cube
        # of the sum.
        perm = torch.arange(x1.shape[0], device=x1.device)
        for rows in row_blocks(groups, x0.shape[0], x1.device):
            if rows.numel() < 2:
                continue
            _, col = linear_sum_assignment(cost[rows][:, rows].detach().cpu().numpy())
            perm[rows] = rows[torch.as_tensor(col, device=x1.device, dtype=torch.long)]
        return x0, x1[perm]


class PCVarianceCoupling(Coupling):
    """
    Rescale the prior draw's principal components so its variance ellipsoid
    matches the data's (both sorted descending); the orientation stays random.

    Changes x1's marginal from the data's values, so the Gaussian kernel is
    lost and sampling needs an explicit
    :class:`~schnetpack.generative.priors.Prior` matching the trained
    statistics. Works on one point cloud at a time (no ``groups``).
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: floor added to both variances before the ratio, guarding
                against a degenerate principal axis
        """
        self.eps = eps

    def pair(self, x0, x1, groups=None):
        if groups is not None:
            # `groups` says which rows may exchange endpoints, which is not the
            # granularity a variance rescale needs: blocking by (molecule,
            # element) would fit an ellipsoid to each element separately.
            # Refusing beats silently rescaling the wrong point set.
            raise NotImplementedError(
                "PCVarianceCoupling has no grouped form yet — it reshapes one "
                "point cloud's variance ellipsoid, so it would need per-molecule "
                "blocks, not the per-(molecule, element) blocks `groups` carries. "
                "Use it without groups, on one structure at a time."
            )
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
    Minibatch optimal-transport pairing (OT flow matching). Not implemented;
    use :class:`PermutationCoupling` for the single-batch special case.
    """

    preserves_marginal = True

    def pair(self, x0, x1, groups=None):
        raise NotImplementedError(
            "OTCoupling lands with the optimal-transport milestone. "
            "Use IdentityCoupling for standard flow matching."
        )

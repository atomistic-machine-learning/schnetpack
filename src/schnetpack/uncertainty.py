"""How far the models of an ensemble disagree.

An ensemble of models predicts a property several times over; the spread of those
predictions is the ensemble's estimate of its own uncertainty. Uncertainty classes
turn the spread into an uncertaintly value as a weighted sum of multiple property
uncertainties.

Predictions arrive with the **model axis first**, already converted to the units the
caller reports in, and keyed by the canonical names of :mod:`schnetpack.properties`::
    properties.energy   (n_models, n_structures)
    properties.forces   (n_models, n_total_atoms, 3)
    properties.stress   (n_models, n_structures, 3, 3)

An absolute uncertainty comes back in those same units, and the result is one value per
structure. Pass ``n_atoms`` whenever the batch holds more than one structure, so the
per-atom terms can be reduced within each structure rather than across the whole batch.

Note:
    The spread is a standard deviation over the models, taken with ``correction=0`` --
    the population convention ``numpy.std`` uses, so that an ensemble reports the same
    number whichever calculator it is driven by.
"""

from abc import ABC, abstractmethod

import torch

from schnetpack import properties
from schnetpack.nn import scatter_add

__all__ = ["Uncertainty", "AbsoluteUncertainty", "RelativeUncertainty"]

#: added to a denominator that a relative uncertainty divides by
_EPS = 1e-8


def _ensemble_std(values: torch.Tensor) -> torch.Tensor:
    """Spread of the predictions over the model axis.

    ``correction=0`` rather than torch's default of 1: a single-model ensemble then has
    an uncertainty of zero rather than ``nan``, and the numbers match ``numpy.std``.
    """
    return values.std(dim=0, correction=0)


def _per_structure_mean(
    per_atom: torch.Tensor, n_atoms: torch.Tensor | None
) -> torch.Tensor:
    """Mean of a per-atom quantity within each structure.

    Args:
        per_atom: ``(n_total_atoms,)``, the atoms of every structure end to end.
        n_atoms: ``(n_structures,)`` atom counts. ``None`` means the whole thing is one
            structure.

    Returns:
        ``(n_structures,)``
    """
    if n_atoms is None:
        return per_atom.mean().reshape(1)

    n_atoms = n_atoms.to(device=per_atom.device)
    idx_m = torch.repeat_interleave(
        torch.arange(n_atoms.shape[0], device=per_atom.device), n_atoms
    )
    totals = scatter_add(per_atom, idx_m, dim_size=n_atoms.shape[0])
    return totals / n_atoms.to(per_atom.dtype)


class Uncertainty(ABC):
    """One number per structure, from the disagreement between the models of an ensemble.

    Each term reduces its property to a *spread* and a *magnitude* of the same shape --
    the energy to the standard deviation over the models and the absolute value of
    their mean; the forces per atom, and the stress per plane, to the length of the
    per-component standard deviation and the length of the mean -- and :meth:`measure`
    turns the pair into a number. The forces are then averaged over a structure's atoms
    and the stress over its three planes, so every term is a scalar per structure before
    the terms are weighted and summed.

    Args:
        energy_weight, force_weight, stress_weight: how much each term contributes. They
            are normalized to sum to one, so only their ratios matter, and a term with
            weight zero is not computed at all -- which is also how a property the
            ensemble does not predict is left out.
    """

    def __init__(
        self,
        energy_weight: float = 0.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
    ):
        total_weight = energy_weight + force_weight + stress_weight
        if total_weight == 0:
            raise ValueError("total_weight cannot be zero")

        self.energy_weight = energy_weight / total_weight
        self.force_weight = force_weight / total_weight
        self.stress_weight = stress_weight / total_weight

    @abstractmethod
    def measure(self, spread: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        """One term's uncertainty from its spread and the magnitude of its mean.

        Args:
            spread: spread over the models, per structure, atom or plane.
            magnitude: length of the models' mean, shaped like ``spread``.
        """

    def __call__(
        self,
        predictions: dict[str, torch.Tensor],
        n_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The uncertainty of every structure the predictions describe.

        Args:
            predictions: the ensemble's predictions, model axis first, in the units the
                result should be reported in, keyed by the names of
                :mod:`schnetpack.properties`.
            n_atoms: ``(n_structures,)`` atom counts, or ``None`` for a single structure.

        Returns:
            ``(n_structures,)``
        """
        terms = []
        if self.energy_weight > 0:
            energy = self._get(predictions, properties.energy, "energy_weight")
            term = self.measure(_ensemble_std(energy), energy.mean(dim=0).abs())
            terms.append(self.energy_weight * term)

        if self.force_weight > 0:
            forces = self._get(predictions, properties.forces, "force_weight")
            per_atom = self._vector_term(forces)
            terms.append(self.force_weight * _per_structure_mean(per_atom, n_atoms))

        if self.stress_weight > 0:
            stress = self._get(predictions, properties.stress, "stress_weight")
            per_plane = self._vector_term(stress)
            terms.append(self.stress_weight * per_plane.mean(dim=-1))

        return sum(terms).reshape(-1)

    def _vector_term(self, values: torch.Tensor) -> torch.Tensor:
        """The measure of every row of the last axis: an atom's force, a stress plane."""
        spread = torch.linalg.norm(_ensemble_std(values), dim=-1)
        magnitude = torch.linalg.norm(values.mean(dim=0), dim=-1)
        return self.measure(spread, magnitude)

    def _get(
        self, predictions: dict[str, torch.Tensor], key: str, weight: str
    ) -> torch.Tensor:
        if key not in predictions:
            raise KeyError(
                f"{type(self).__name__} has a nonzero {weight}, but the ensemble "
                f"predicts no {key!r}; set the weight to zero or predict it"
            )
        return predictions[key]


class AbsoluteUncertainty(Uncertainty):
    """The spread itself, in the units of the properties it is built from."""

    def measure(self, spread: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        return spread


class RelativeUncertainty(Uncertainty):
    """The spread as a fraction of the predicted magnitude, so it is dimensionless.

    Useful when the structures of a batch differ enough in size or composition that their
    absolute uncertainties are not comparable. It grows as the predicted magnitude goes
    to zero -- the forces of a relaxed structure, say -- even where the spread does not.
    """

    def measure(self, spread: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        return spread / (magnitude + _EPS)

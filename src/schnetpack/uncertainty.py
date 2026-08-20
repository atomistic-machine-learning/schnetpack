"""How far the members of an ensemble disagree.

An ensemble of models predicts a property several times over; the spread of those
predictions is the ensemble's estimate of its own uncertainty. Turning that spread into
one comparable number per structure is what the classes here do, and how the energy, the
forces and the stress are weighted against each other is the caller's choice.

Predictions arrive with the **model axis first** and already converted to the units the
caller reports in, so an absolute uncertainty comes back in those same units::

    energy   (n_models, n_structures)
    forces   (n_models, n_total_atoms, 3)
    stress   (n_models, n_structures, 3, 3)

and the result is one value per structure. Pass ``n_atoms`` whenever the batch holds more
than one structure, so the per-atom terms can be reduced within each structure rather than
across the whole batch.

These live here rather than next to a calculator because both
:class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`, which works on one
``ase.Atoms`` at a time, and
:class:`~schnetpack.relax.BatchwiseEnsembleCalculator`, which works on a whole batch of
tensors and must stay free of ase, use the same ones.

Note:
    The spread is a standard deviation over the models, taken with ``correction=0`` --
    the population convention ``numpy.std`` uses, so that an ensemble reports the same
    number whichever calculator it is driven by.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

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
    per_atom: torch.Tensor, n_atoms: Optional[torch.Tensor]
) -> torch.Tensor:
    """Mean of a per-atom quantity within each structure.

    Args:
        per_atom: ``(n_total_atoms,)``, the atoms of every structure end to end.
        n_atoms: ``(n_structures,)`` atom counts. ``None`` means the whole thing is one
            structure, which is the case an ase calculator is in.

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
    """One number per structure, from the disagreement between ensemble members.

    Args:
        energy_key, force_key, stress_key: names the predictions are keyed by.
        energy_weight, force_weight, stress_weight: how much each term contributes. They
            are normalized to sum to one, so only their ratios matter, and a term with
            weight zero is not computed at all -- which is also how a property the
            ensemble does not predict is left out.
    """

    def __init__(
        self,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: str = "stress",
        energy_weight: float = 0.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
    ):
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # normalize weights
        total_weight = energy_weight + force_weight + stress_weight
        if total_weight == 0:
            raise ValueError("total_weight cannot be zero")

        self.energy_weight = energy_weight / total_weight
        self.force_weight = force_weight / total_weight
        self.stress_weight = stress_weight / total_weight

    @abstractmethod
    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        n_atoms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The uncertainty of every structure the predictions describe.

        Args:
            predictions: the ensemble's predictions, model axis first, in the units the
                result should be reported in.
            n_atoms: ``(n_structures,)`` atom counts, or ``None`` for a single structure.

        Returns:
            ``(n_structures,)``
        """


class AbsoluteUncertainty(Uncertainty):
    """The spread itself, in the units of the properties it is built from.

    The force term is the mean over a structure's atoms of the length of the per-atom
    standard deviation vector, and the stress term the mean over the three planes of the
    same thing -- so all three terms are scalars per structure before they are weighted.
    """

    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        n_atoms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        uncertainty = None

        if self.energy_weight > 0:
            energy_unc = _ensemble_std(predictions[self.energy_key])
            uncertainty = self.energy_weight * energy_unc

        if self.force_weight > 0:
            # per atom uncertainty as the L2 norm of the per-component stds
            force_std = _ensemble_std(predictions[self.force_key])
            per_atom_uncertainty = torch.linalg.norm(force_std, dim=-1)
            force_unc = _per_structure_mean(per_atom_uncertainty, n_atoms)
            term = self.force_weight * force_unc
            uncertainty = term if uncertainty is None else uncertainty + term

        if self.stress_weight > 0:
            # uncertainty per plane, then averaged over the three of them
            stress_std = _ensemble_std(predictions[self.stress_key])
            per_plane_uncertainty = torch.linalg.norm(stress_std, dim=-1)
            stress_unc = per_plane_uncertainty.mean(dim=-1)
            term = self.stress_weight * stress_unc
            uncertainty = term if uncertainty is None else uncertainty + term

        return uncertainty.reshape(-1)


class RelativeUncertainty(Uncertainty):
    """The spread as a fraction of the predicted magnitude, so it is dimensionless.

    Useful when the structures of a batch differ enough in size or composition that their
    absolute uncertainties are not comparable.
    """

    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        n_atoms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        uncertainty = None

        if self.energy_weight > 0:
            energy_preds = predictions[self.energy_key]
            mean_energy = energy_preds.mean(dim=0)
            std_energy = _ensemble_std(energy_preds)
            energy_unc = std_energy / (mean_energy.abs() + _EPS)
            uncertainty = self.energy_weight * energy_unc

        if self.force_weight > 0:
            force_preds = predictions[self.force_key]
            mean_norms = torch.linalg.norm(force_preds.mean(dim=0), dim=-1)
            std_norms = torch.linalg.norm(_ensemble_std(force_preds), dim=-1)
            per_atom = std_norms / (mean_norms + _EPS)
            term = self.force_weight * _per_structure_mean(per_atom, n_atoms)
            uncertainty = term if uncertainty is None else uncertainty + term

        if self.stress_weight > 0:
            stress_preds = predictions[self.stress_key]
            mean_planes = torch.linalg.norm(stress_preds.mean(dim=0), dim=-1)
            std_planes = torch.linalg.norm(_ensemble_std(stress_preds), dim=-1)
            per_plane = std_planes / (mean_planes + _EPS)
            term = self.stress_weight * per_plane.mean(dim=-1)
            uncertainty = term if uncertainty is None else uncertainty + term

        return uncertainty.reshape(-1)

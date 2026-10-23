"""
Guidance: terms added to the field a driver steps on.

Where a :class:`~schnetpack.dynamics.hooks.Hook` edits the iterate between
a driver's steps, guidance changes the field itself — restraint forces,
classifier guidance — so it acts at every point the field is evaluated
(Heun's predictor, an L-BFGS step alike), and the driver never sees it apart
from the model's own field.

A guidance term is a :class:`~torch.nn.Module` whose ``forward`` reads what
it needs from the batch and returns a tensor shaped like the field it is
added to. Its ``weight`` scales it, and whoever holds the term adds the
weighted term to its field:

- :class:`~schnetpack.dynamics.calculator.ForceCalculator` — to the model's
  forces.
- an :class:`~schnetpack.dynamics.sample.sampler.Update` of a
  :class:`~schnetpack.dynamics.sample.sampler.Sampler` — to the field the
  update steps on, of whichever variable it steps: the score of an SDE
  update, the velocity of an ODE update, the x0 of an ancestral one. The
  batch the term sees holds the stage point and the path time per atom and
  per structure, for guidance that depends on t.

The weight carries a term into the unit of its field.
:class:`HarmonicRestraint`, a force on the positions in eV/Angstrom, enters a
physical force with a plain factor, a pseudo-force with Angstrom^2/eV, and a
score with 1/kT, in 1/eV.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from schnetpack import properties

__all__ = ["Guidance", "HarmonicRestraint"]


class Guidance(nn.Module):
    """
    Base class of the guidance terms.

    Subclasses implement :meth:`forward`. Unlike hooks their
    order does not matter: every one is evaluated on the same batch, and the
    weighted terms of all of them add up.
    """

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: factor this term is scaled by; carries it into the unit
                of the field it is added to
        """
        super().__init__()
        self.weight = weight

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the term at ``batch``.

        The batch is the step rule's evaluation point, not necessarily the
        driver's iterate. In sampling it also holds the path time per atom
        and per structure; a term that needs it should raise if it is
        missing.

        Args:
            batch: batch to evaluate on

        Returns:
            The term, unweighted and shaped like the field it is added to —
            the calculator or the update applies :attr:`weight`. ``batch``
            is left unchanged.
        """
        raise NotImplementedError


class HarmonicRestraint(Guidance):
    """
    Harmonic restraints on the distances of atom pairs,

        E = 1/2 k (d - d0)^2,

    summed per structure. Pairs, target distances and force constants are
    batch keys, so they collate with the structures and every structure may
    carry its own restraints — or none. By default they are read from

    - ``batch[HarmonicRestraint.restraint_pairs]`` — ``(n_pairs, 2)`` atom
      indices, local to their structure, the pairs of all structures end to
      end
    - ``batch[HarmonicRestraint.n_restraints]`` — ``(n_structures,)`` number
      of pairs per structure
    - ``batch[HarmonicRestraint.restraint_lengths]`` — ``(n_pairs,)`` target
      distances d0 in Angstrom
    - ``batch[HarmonicRestraint.restraint_constants]`` — ``(n_pairs,)`` force
      constants k in eV/Angstrom^2

    :meth:`forward` returns the forces -dE/dR ``(n_atoms, 3)`` in
    eV/Angstrom, at the positions ``batch[properties.R]`` in Angstrom;
    :meth:`energy` gives E itself.
    """

    restraint_pairs = "_restraint_pairs"  #: default key of the restrained pairs
    n_restraints = "_n_restraints"  #: default key of the pairs per structure
    restraint_lengths = "_restraint_lengths"  #: default key of the target distances
    restraint_constants = "_restraint_constants"  #: default key of the force constants

    def __init__(
        self,
        pairs_key: str = restraint_pairs,
        count_key: str = n_restraints,
        lengths_key: str = restraint_lengths,
        constants_key: str = restraint_constants,
        weight: float = 1.0,
    ):
        """
        Args:
            pairs_key: batch key of the restrained pairs
            count_key: batch key of the number of pairs per structure
            lengths_key: batch key of the target distances, in Angstrom
            constants_key: batch key of the force constants, in eV/Angstrom^2
            weight: factor the forces are scaled by: a plain factor on a
                physical force, Angstrom^2/eV on a pseudo-force, 1/kT in
                1/eV on a score
        """
        super().__init__(weight=weight)
        self.pairs_key = pairs_key
        self.count_key = count_key
        self.lengths_key = lengths_key
        self.constants_key = constants_key

    def _stretches(self, batch):
        """
        The restrained pairs of ``batch`` and how far each is stretched.

        Returns:
            ``idx`` ``(n_pairs, 2)`` batch indices of the pairs, ``idx_pair_m``
            ``(n_pairs,)`` their structures, ``r_ij`` ``(n_pairs, 3)``,
            ``d`` and ``stretch = d - d0`` ``(n_pairs,)`` and the force
            constants ``(n_pairs,)``.
        """
        positions = batch[properties.R]
        n_atoms = batch[properties.n_atoms].to(positions.device)
        count = batch[self.count_key].to(device=positions.device, dtype=torch.long)
        pairs = batch[self.pairs_key].to(device=positions.device, dtype=torch.long)
        lengths = batch[self.lengths_key].to(positions)
        constants = batch[self.constants_key].to(positions)
        if count.shape != n_atoms.shape:
            raise ValueError(
                f"{self.count_key!r} must hold one count per structure: shape "
                f"{tuple(n_atoms.shape)}, got {tuple(count.shape)}"
            )
        n_pairs = int(count.sum())
        for key, value, shape in (
            (self.pairs_key, pairs, (n_pairs, 2)),
            (self.lengths_key, lengths, (n_pairs,)),
            (self.constants_key, constants, (n_pairs,)),
        ):
            if value.shape != shape:
                raise ValueError(
                    f"{key!r} must be shaped {shape} for {n_pairs} restrained "
                    f"pairs, got {tuple(value.shape)}"
                )

        # local pair indices -> batch indices, by the first atom of the
        # structure each pair belongs to
        first = torch.cumsum(n_atoms, dim=0) - n_atoms
        idx_pair_m = torch.repeat_interleave(
            torch.arange(n_atoms.shape[0], device=positions.device), count
        )
        idx = pairs + first[idx_pair_m].unsqueeze(-1)

        r_ij = positions[idx[:, 1]] - positions[idx[:, 0]]
        d = torch.norm(r_ij, dim=-1)
        return idx, idx_pair_m, r_ij, d, d - lengths, constants

    def energy(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The restraint energy per structure ``(n_structures,)`` in eV, at
        ``batch[properties.R]``, unweighted.
        """
        _, idx_pair_m, _, _, stretch, constants = self._stretches(batch)
        positions = batch[properties.R]
        return torch.zeros(
            batch[properties.n_atoms].shape[0],
            dtype=positions.dtype,
            device=positions.device,
        ).index_add_(0, idx_pair_m, 0.5 * constants * stretch**2)

    def forward(self, batch):
        idx, _, r_ij, d, stretch, constants = self._stretches(batch)
        # F_j = -k (d - d0) r_ij / d, F_i = -F_j
        f_j = -(constants * stretch / d).unsqueeze(-1) * r_ij
        forces = torch.zeros_like(batch[properties.R])
        forces.index_add_(0, idx[:, 1], f_j)
        forces.index_add_(0, idx[:, 0], -f_j)
        return forces


def guidance_terms(terms: Sequence) -> list[Guidance]:
    """``terms`` as a list, after checking that each is a :class:`Guidance`."""
    terms = list(terms)
    for term in terms:
        if not isinstance(term, Guidance):
            raise TypeError(f"{type(term).__name__} is not a Guidance")
    return terms


def weighted_sum(terms: Sequence[Guidance], batch: Mapping[str, Any]) -> torch.Tensor:
    """The sum of at least one term at ``batch``, each scaled by its ``weight``."""
    return sum(term.weight * term(batch) for term in terms)

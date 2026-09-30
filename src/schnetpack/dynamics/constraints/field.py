"""
Field-level constraints: terms added to the field a
:class:`~schnetpack.dynamics.base.Dynamics` step follows.

Where a :class:`~schnetpack.dynamics.constraints.state.StateConstraint` edits
the iterate between steps, a field constraint adds to what the step is driven
by — restraint forces, guidance — so its effect enters through the integrator
rather than around it, at every point the integrator evaluates the field
(Heun's predictor, an L-BFGS step alike).

A field constraint returns one force-like term. It is a
:class:`~torch.nn.Module` whose ``forward`` reads the positions in
``batch[properties.R]``, in Angstrom, and returns a tensor ``(n_atoms, 3)`` in
eV/Angstrom: the force of a restraint, or kT times the gradient of a
log-density for guidance that is no energy (a classifier's
kT ∇ log p(y|x)). No energy is returned, so a term need not be conservative.
Its ``weight`` scales it, and the driver applies that weight. Folding the
weighted terms into the field is the driver's job, since only the driver knows
what its field is:

- :class:`~schnetpack.dynamics.relax.Relaxer` — the term is part of the
  force field: it is added to the model's forces, and the weight is a plain
  factor.
- :class:`~schnetpack.dynamics.sampling.Sampler` — guidance: the weighted term
  is added to the score, so the weight is 1/kT, in 1/eV. The batch the
  constraint sees then also holds the path time under the sampler's
  ``time_key``, for guidance that depends on t.
"""

import torch
from torch import nn

from schnetpack import properties

__all__ = ["FieldConstraint", "HarmonicRestraint"]


class FieldConstraint(nn.Module):
    """
    Base class of field-level constraints.

    Subclasses implement :meth:`forward`. Unlike state constraints their
    order does not matter: every one is evaluated on the same batch, and the
    weighted terms of all field constraints add up.
    """

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: factor the driver scales this constraint's term by: a
                plain factor in a relaxer, 1/kT in 1/eV in a sampler
        """
        super().__init__()
        self.weight = weight

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the constraint at ``batch[properties.R]``.

        The positions are in Angstrom and are the integrator's evaluation
        point, not necessarily the driver's iterate. During sampling the batch
        also holds the path time under the sampler's ``time_key``; a
        constraint that needs it should raise if it is missing.

        Args:
            batch: batch to evaluate on, for the positions, the structure
                layout and the constraint's own inputs

        Returns:
            The force-like term ``(n_atoms, 3)`` in eV/Angstrom, unweighted —
            the driver applies :attr:`weight`. ``batch`` is left unchanged.
        """
        raise NotImplementedError


class HarmonicRestraint(FieldConstraint):
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

    :meth:`forward` returns the forces -dE/dR; :meth:`energy` gives E itself.
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
            weight: factor the driver scales the forces by: a plain factor
                in a relaxer, 1/kT in 1/eV in a sampler
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

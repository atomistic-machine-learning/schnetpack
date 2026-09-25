"""
Field-level constraints: changes to the field a step is driven by (restraint
forces, guidance), entering through the integrator rather than around it.
"""

__all__ = ["FieldConstraint"]


# TODO: nothing calls `modify_field` yet. Wire it into the drivers (the
# Sampler's reverse field and DirectDenoising's jump) so restraint forces and
# guidance can act through the integrator, and reconcile with the energy-term
# restraints (HarmonicBond) on the jl/optimizer_performance_v2 branch.
class FieldConstraint:
    """Base class of field-level constraints; the hook defaults to identity."""

    def modify_field(self, batch, field, dynamics):
        """
        Return the, possibly modified, field for the current step.

        Args:
            batch: current batch
            field: field the step would follow, shaped like the moved key
            dynamics: the running :class:`~schnetpack.dynamics.base.Dynamics`
        """
        return field

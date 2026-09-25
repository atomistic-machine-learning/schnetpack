"""
Field-level constraints: changes to the field a step is driven by (restraint
forces, guidance), entering through the integrator rather than around it.
"""

__all__ = ["FieldConstraint"]


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

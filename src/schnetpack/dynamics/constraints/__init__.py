"""
State constraints on a :class:`~schnetpack.dynamics.base.Dynamics` loop:
edits of the iterate between steps (:class:`StateConstraint`,
:class:`Scaffold`). Terms that change the field itself are
:mod:`~schnetpack.dynamics.guidance`, given to the calculator.
"""

from schnetpack.dynamics.constraints.state import *

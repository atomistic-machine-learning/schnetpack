"""
Deprecated location of the Lightning training task.

``AtomisticTask`` moved to :mod:`schnetpack.lightning`; the loss helpers live
in :mod:`schnetpack.objectives`. They are re-exported here so that configs and
checkpoints referencing ``schnetpack.task.ModelOutput`` keep working.
"""

import warnings

from schnetpack.objectives import (
    ConsiderOnlySelectedAtoms,
    ModelOutput,
    UnsupervisedModelOutput,
)

__all__ = ["ModelOutput", "UnsupervisedModelOutput", "ConsiderOnlySelectedAtoms"]


def __getattr__(name):
    if name == "AtomisticTask":
        warnings.warn(
            "`schnetpack.task.AtomisticTask` moved to "
            "`schnetpack.lightning.AtomisticTask`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from schnetpack.lightning import AtomisticTask

        return AtomisticTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

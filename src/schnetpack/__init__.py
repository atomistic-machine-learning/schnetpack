import importlib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboard")

from schnetpack import (
    data,
    datasets,
    dynamics,
    generative,
    interfaces,
    md,
    model,
    nn,
    objectives,
    properties,
    train,
    transform,
    uncertainty,
)
from schnetpack.objectives import (
    AtomMask,
    LossMask,
    ModelOutput,
    UnsupervisedModelOutput,
)
from schnetpack.units import *

__version__ = "3.0.0"


def __getattr__(name):
    # `schnetpack.lightning` is not imported eagerly so that PyTorch Lightning
    # is only loaded when it is used.
    if name in ("lightning", "task"):
        return importlib.import_module(f"schnetpack.{name}")
    if name == "AtomisticTask":
        warnings.warn(
            "`schnetpack.AtomisticTask` moved to `schnetpack.lightning.AtomisticTask`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module("schnetpack.lightning").AtomisticTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

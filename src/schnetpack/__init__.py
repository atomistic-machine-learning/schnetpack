import importlib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboard")

from schnetpack import transform
from schnetpack import properties
from schnetpack import data
from schnetpack import datasets
from schnetpack import interfaces
from schnetpack import nn
from schnetpack import train
from schnetpack import model
from schnetpack import objectives
from schnetpack import generative
from schnetpack import dynamics
from schnetpack.units import *
from schnetpack.objectives import (
    ConsiderOnlySelectedAtoms,
    ModelOutput,
    UnsupervisedModelOutput,
)
from schnetpack import md

__version__ = "2.2.0"


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

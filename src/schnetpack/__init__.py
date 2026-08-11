import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboard")

from schnetpack import (
    atomistic,
    data,
    datasets,
    interfaces,
    md,
    model,
    nn,
    properties,
    representation,
    train,
    transform,
)
from schnetpack.task import *
from schnetpack.units import *

__version__ = "2.2.0"

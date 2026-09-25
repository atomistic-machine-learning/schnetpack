"""
PyTorch Lightning integration: training task, datamodule and callbacks.

Everything in SchNetPack that depends on PyTorch Lightning lives here. The
models, data pipeline and :mod:`schnetpack.objectives` work without it, so
custom PyTorch training loops only need this package if they opt in.
"""

from .task import *
from .datamodule import *
from .callbacks import *

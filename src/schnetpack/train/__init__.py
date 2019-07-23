r"""
Classes to manage the training process.

schnetpack.train.Trainer encapsulates the training loop. It also can automatically monitor the performance on the
validation set and contains logic for checkpointing. The training process can be customized using Hooks which derive
from schnetpack.train.Hooks.

"""

from .trainer import Trainer
from .loss import *
from .hooks import *
from .metrics import *

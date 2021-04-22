import logging

import hydra
import pytorch_lightning.metrics
import torch

import schnetpack as spk
import cProfile

from schnetpack.model.base import AtomisticModel

log = logging.getLogger(__name__)


class PESModel(AtomisticModel):
    pass

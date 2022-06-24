from copy import copy
from typing import Dict

import torch
import os
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import List, Any
from schnetpack.task import AtomisticTask

__all__ = ["PredictionWriter"]


class PredictionWriter(BasePredictionWriter):
    """
    Callback to store prediction results using ``torch.save``.
    """

    def __init__(self, output_dir: str, write_interval: str):
        """
        Args:
            output_dir: output directory for prediction files
            write_interval: can be one of ["batch", "epoch", "batch_and_epoch"]
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module: AtomisticTask,
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        bdir = os.path.join(self.output_dir, str(dataloader_idx))
        os.makedirs(bdir, exist_ok=True)
        torch.save(prediction, os.path.join(bdir, f"{batch_idx}.pt"))

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: AtomisticTask,
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))

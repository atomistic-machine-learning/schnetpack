from copy import copy
from typing import Dict

from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint

import torch
import os
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import List, Any
from schnetpack.atomistic import AtomisticModel

__all__ = ["ModelCheckpoint", "PredictionWriter"]


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
        pl_module: AtomisticModel,
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
        pl_module: AtomisticModel,
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


class ModelCheckpoint(BaseModelCheckpoint):
    """
    Like the PyTorch Lightning ModelCheckpoint callback,
    but also saves the best inference model with activated post-processing
    """

    def __init__(self, inference_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_path = inference_path

    def on_validation_end(self, trainer, pl_module) -> None:
        self.trainer = trainer
        self.pl_module = pl_module
        super().on_validation_end(trainer, pl_module)

    def _update_best_and_save(
        self, current: torch.Tensor, trainer, monitor_candidates: Dict[str, Any]
    ):
        # save model checkpoint
        super()._update_best_and_save(current, trainer, monitor_candidates)

        # save best inference model
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"))

        if current == self.best_model_score:

            if self.trainer.training_type_plugin.should_rank_save_checkpoint:
                # remove references to trainer and data loaders to avoid pickle error in ddp
                model = copy(self.pl_module)
                model.eval()
                model.inference_mode = True
                model.trainer = None
                model.train_dataloader = None
                model.val_dataloader = None
                model.test_dataloader = None
                torch.save(model, self.inference_path)
                model.train()

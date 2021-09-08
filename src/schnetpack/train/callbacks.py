from typing import Dict, Any

import torch

from pytorch_lightning.callbacks import ModelCheckpoint
import schnetpack as spk
from pytorch_lightning import Trainer
from copy import copy

__all__ = ["ModelCheckpoint"]


class ModelCheckpoint(ModelCheckpoint):
    """
    Just like the PyTorch Lightning ModelCheckpoint callback,
    but also saves the best inference model with activated post-processing
    """

    def __init__(self, inference_path: str, method: str = "script", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_path = inference_path
        self.method = method

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
            if isinstance(self.pl_module, spk.atomistic.model.AtomisticModel):
                imode = self.pl_module.inference_mode
                self.pl_module.inference_mode = True
            mode = self.pl_module.training

            if self.trainer.training_type_plugin.should_rank_save_checkpoint:
                # remove references to trainer and data loaders to avoid picle error in ddp
                model = copy(self.pl_module)
                model.trainer = None
                model.train_dataloader = None
                model.val_dataloader = None
                model.test_dataloader = None
                torch.save(model, self.inference_path)

            if isinstance(self.pl_module, spk.atomistic.model.AtomisticModel):
                self.pl_module.inference_mode = imode
            self.pl_module.train(mode)

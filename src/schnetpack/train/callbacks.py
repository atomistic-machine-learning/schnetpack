from typing import Dict, Any

import torch

import schnetpack as spk
from pytorch_lightning.callbacks import ModelCheckpoint

__all__ = ["ModelCheckpoint"]


class ModelCheckpoint(ModelCheckpoint):
    """
    Just like the PyTorch Lightning ModelCheckpoint callback,
    but also saves the best inference model
    """

    def __init__(
        self,
        inference_path: str,
        save_as_torch_script: bool = True,
        method: str = "script",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_as_torch_script = save_as_torch_script
        self.inference_path = inference_path
        self.method = method

    def on_validation_end(self, trainer, pl_module) -> None:
        self.pl_module = pl_module
        super().on_validation_end(trainer, pl_module)

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        epoch: int,
        step: int,
        trainer,
        monitor_candidates: Dict[str, Any],
    ):
        # save model checkpoint
        super()._update_best_and_save(current, epoch, step, trainer, monitor_candidates)

        # save best inference model
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"))

        if current == self.best_model_score:
            if self.save_as_torch_script:
                self.pl_module.to_torchscript(self.inference_path, method=self.method)
            else:
                if isinstance(self.pl_module, spk.model.AtomisticModel):
                    imode = self.pl_module.inference_mode
                    self.pl_module.inference_mode = True
                mode = self.pl_module.training

                torch.save(self.pl_module, self.inference_path)

                if isinstance(self.pl_module, spk.model.AtomisticModel):
                    self.pl_module.inference_mode = imode
                self.pl_module.train(mode)

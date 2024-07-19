from copy import copy
from typing import Dict

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint

from torch_ema import ExponentialMovingAverage as EMA

import torch
import os
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import List, Any
from schnetpack.task import AtomisticTask
from schnetpack import properties
from collections import defaultdict


__all__ = ["ModelCheckpoint", "PredictionWriter", "ExponentialMovingAverage"]


class PredictionWriter(BasePredictionWriter):
    """
    Callback to store prediction results using ``torch.save``.
    """

    def __init__(
        self,
        output_dir: str,
        write_interval: str,
        write_idx: bool = False,
    ):
        """
        Args:
            output_dir: output directory for prediction files
            write_interval: can be one of ["batch", "epoch", "batch_and_epoch"]
            write_idx: Write molecular ids for all atoms. This is needed for
                atomic properties like forces.
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.write_idx = write_idx
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
        # collect batches of predictions and restructure
        concatenated_predictions = defaultdict(list)
        for batch_prediction in predictions[0]:
            for property_name, data in batch_prediction.items():
                if not self.write_idx and property_name == properties.idx_m:
                    continue
                concatenated_predictions[property_name].append(data)
        concatenated_predictions = {
            property_name: torch.concat(data)
            for property_name, data in concatenated_predictions.items()
        }

        # save concatenated predictions
        torch.save(
            concatenated_predictions,
            os.path.join(self.output_dir, "predictions.pt"),
        )


class ModelCheckpoint(BaseModelCheckpoint):
    """
    Like the PyTorch Lightning ModelCheckpoint callback,
    but also saves the best inference model with activated post-processing
    """

    def __init__(self, model_path: str, do_postprocessing=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.do_postprocessing = do_postprocessing

    def on_validation_end(self, trainer, pl_module: AtomisticTask) -> None:
        self.trainer = trainer
        self.task = pl_module
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
            if self.trainer.strategy.local_rank == 0:
                # remove references to trainer and data loaders to avoid pickle error in ddp
                self.task.save_model(self.model_path, do_postprocessing=True)


class ExponentialMovingAverage(Callback):
    def __init__(self, decay, *args, **kwargs):
        self.decay = decay
        self.ema = None
        self._to_load = None

    def on_fit_start(self, trainer, pl_module: AtomisticTask):
        if self.ema is None:
            self.ema = EMA(pl_module.model.parameters(), decay=self.decay)
        if self._to_load is not None:
            self.ema.load_state_dict(self._to_load)
            self._to_load = None

        # load average parameters, to have same starting point as after validation
        self.ema.store()
        self.ema.copy_to()

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ema.restore()

    def on_train_batch_end(self, trainer, pl_module: AtomisticTask, *args, **kwargs):
        self.ema.update()

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: AtomisticTask, *args, **kwargs
    ):
        self.ema.store()
        self.ema.copy_to()

    def load_state_dict(self, state_dict):
        if "ema" in state_dict:
            if self.ema is None:
                self._to_load = state_dict["ema"]
            else:
                self.ema.load_state_dict(state_dict["ema"])

    def state_dict(self):
        return {"ema": self.ema.state_dict()}

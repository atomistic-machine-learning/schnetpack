import os
from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint
from torch_ema import ExponentialMovingAverage as EMA

from schnetpack import properties
from schnetpack.task import AtomisticTask

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
        batch_indices: list[int],
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
        predictions: list[Any],
        batch_indices: list[Any],
    ):
        # Lightning passes a flat list of batch predictions for a single dataloader,
        # and one such list per dataloader otherwise (only the first one is written)
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]

        # collect batches of predictions and restructure
        concatenated_predictions = defaultdict(list)
        for batch_prediction in predictions:
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
        self._last_saved_step = None

    def on_train_epoch_end(self, trainer, pl_module: AtomisticTask) -> None:
        skip = self._should_skip_saving_checkpoint(trainer)
        super().on_train_epoch_end(trainer, pl_module)
        if not skip and self._should_save_on_train_epoch_end(trainer):
            self._ensure_last_checkpoint(trainer)

    def on_validation_end(self, trainer, pl_module: AtomisticTask) -> None:
        self.trainer = trainer
        self.task = pl_module
        skip = self._should_skip_saving_checkpoint(trainer)
        super().on_validation_end(trainer, pl_module)
        if not skip and not self._should_save_on_train_epoch_end(trainer):
            self._ensure_last_checkpoint(trainer)

    def _save_last_checkpoint(self, trainer, monitor_candidates: dict[str, Any]):
        super()._save_last_checkpoint(trainer, monitor_candidates)
        self._last_saved_step = trainer.global_step

    def _ensure_last_checkpoint(self, trainer):
        # Since PyTorch Lightning 2.6.0 the last checkpoint is only written in
        # steps that also save a top-k checkpoint
        # (Lightning-AI/pytorch-lightning#22043), so once the monitored metric
        # stops improving, resuming from it would repeat epochs. Write it here
        # whenever Lightning did not, as Lightning < 2.6 does.
        if self.save_last and self._last_saved_step != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def _update_best_and_save(
        self, current: torch.Tensor, trainer, monitor_candidates: dict[str, Any]
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
    """
    Keeps an exponential moving average (EMA) of the model weights.

    Validation, and thereby checkpoints and the best model, use the EMA average,
    while training continues on the raw weights.
    """

    def __init__(self, decay, *args, **kwargs):
        self.decay = decay
        self.ema = None
        self._to_load = None
        # True while the model holds the EMA average and self.ema stores the raw
        # training weights
        self._average_loaded = False

    def on_fit_start(self, trainer, pl_module: AtomisticTask):
        if self.ema is None:
            self.ema = EMA(pl_module.model.parameters(), decay=self.decay)
        if self._to_load is not None:
            self._load_ema_state(*self._to_load)
            self._to_load = None

        # load average parameters, to have same starting point as after validation
        if not self._average_loaded:
            self._load_average()

    def on_train_batch_start(self, trainer, pl_module: AtomisticTask, *args, **kwargs):
        # Validation loads the average; train on the raw weights again. Doing this
        # per batch, not per epoch, also covers validation within an epoch and
        # epochs without validation.
        if self._average_loaded:
            self.ema.restore()
            self._average_loaded = False

    def on_train_batch_end(self, trainer, pl_module: AtomisticTask, *args, **kwargs):
        self.ema.update()

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: AtomisticTask, *args, **kwargs
    ):
        # The average is already loaded during the sanity check and when no
        # training step ran since the last validation. Storing again would
        # overwrite the raw weights with the average.
        if not self._average_loaded:
            self._load_average()

    def _load_average(self):
        self.ema.store()
        self.ema.copy_to()
        self._average_loaded = True

    def _load_ema_state(self, ema_state, average_loaded):
        self.ema.load_state_dict(ema_state)
        if average_loaded is None:
            # checkpoints without the flag were saved with the average loaded
            # whenever raw weights had been stored
            average_loaded = self.ema.collected_params is not None
        self._average_loaded = average_loaded

    def load_state_dict(self, state_dict):
        if "ema" in state_dict:
            to_load = (state_dict["ema"], state_dict.get("average_loaded"))
            if self.ema is None:
                self._to_load = to_load
            else:
                self._load_ema_state(*to_load)

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "average_loaded": self._average_loaded}

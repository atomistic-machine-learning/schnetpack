from copy import copy
from typing import Dict, List, Optional

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

from torchmetrics import MeanAbsoluteError, MeanSquaredError

__all__ = [
    "ModelCheckpoint",
    "PredictionWriter",
    "ExponentialMovingAverage",
    "DatasetMetrics",
]


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


class DatasetMetrics(Callback):
    """
    Computes per-dataset MAE and RMSE for energy and forces at test time.
    """

    def __init__(
        self,
        dataset_names: List[str],
        energy_key: str = "energy",
        forces_key: str = "forces",
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.energy_key = energy_key
        self.forces_key = forces_key

        # metrics are created in setup() so they land on the correct device
        self.energy_mae: Optional[Dict] = None
        self.energy_rmse: Optional[Dict] = None
        self.forces_mae: Optional[Dict] = None
        self.forces_rmse: Optional[Dict] = None

    def _init_metrics(self, device):
        print(f"[DatasetMetrics] _init_metrics called, device={device}")
        self.energy_mae = {
            n: MeanAbsoluteError().to(device) for n in self.dataset_names
        }
        self.energy_rmse = {
            n: MeanSquaredError(squared=False).to(device) for n in self.dataset_names
        }
        self.forces_mae = {
            n: MeanAbsoluteError().to(device) for n in self.dataset_names
        }
        self.forces_rmse = {
            n: MeanSquaredError(squared=False).to(device) for n in self.dataset_names
        }

    def setup(self, trainer, pl_module, stage: str):
        self._init_metrics(pl_module.device)  # always init regardless of stage

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self.energy_mae is None:
            self._init_metrics(pl_module.device)

        idx_m = batch[properties.idx_m]
        dataset_id = batch["dataset_id"].squeeze(-1)
        dataset_id_per_atom = dataset_id[idx_m]

        pred_energy = batch["_pred_energy"]
        true_energy = batch["_true_energy"]
        pred_forces = batch["_pred_forces"]
        true_forces = batch["_true_forces"]

        if batch_idx == 0:
            print(f"pred_energy: {pred_energy[:5]}")
            print(f"true_energy: {true_energy[:5]}")

        for d, name in enumerate(self.dataset_names):
            mol_mask = dataset_id == d
            if mol_mask.any():
                self.energy_mae[name].update(
                    pred_energy[mol_mask], true_energy[mol_mask]
                )
                self.energy_rmse[name].update(
                    pred_energy[mol_mask], true_energy[mol_mask]
                )

            atom_mask = dataset_id_per_atom == d
            if atom_mask.any():
                self.forces_mae[name].update(
                    pred_forces[atom_mask].reshape(-1),
                    true_forces[atom_mask].reshape(-1),
                )
                self.forces_rmse[name].update(
                    pred_forces[atom_mask].reshape(-1),
                    true_forces[atom_mask].reshape(-1),
                )

    def on_test_epoch_end(self, trainer, pl_module):
        for name in self.dataset_names:
            pl_module.log(
                f"test/{name}_energy_mae",
                self.energy_mae[name].compute(),
                sync_dist=False,
            )
            pl_module.log(
                f"test/{name}_energy_rmse",
                self.energy_rmse[name].compute(),
                sync_dist=False,
            )
            pl_module.log(
                f"test/{name}_forces_mae",
                self.forces_mae[name].compute(),
                sync_dist=False,
            )
            pl_module.log(
                f"test/{name}_forces_rmse",
                self.forces_rmse[name].compute(),
                sync_dist=False,
            )

            self.energy_mae[name].reset()
            self.energy_rmse[name].reset()
            self.forces_mae[name].reset()
            self.forces_rmse[name].reset()

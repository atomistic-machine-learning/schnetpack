from copy import copy
from typing import Dict
import numpy as np

from pytorch_lightning import LightningModule, Trainer
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
import h5py

__all__ = ["ModelCheckpoint", "PredictionWriter", "ExponentialMovingAverage","So3kratesCallback"]


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
        from ase import Atoms

        start_idx = 0 
        end_idx = 0
        R = batch[properties.R].detach().cpu().numpy()
        Z = batch[properties.Z].detach().cpu().numpy()

        all_R = []
        all_Z = []
        dihedrals = []
        idx_m = prediction[properties.idx_m].unique().detach().cpu().numpy()
        for i in range(len(idx_m)):
            start = i * 13
            end = (i+1)  * 13
            atm = Atoms(Z[start:end], positions=R[start:end])
            start_idx += i
            end_idx += i
            indices = np.array([1,2,10,11]).reshape(1,-1)
            angle = atm.get_dihedrals(indices=indices)
            dihedrals.append(angle)
            all_R.append(R[start:end])
            all_Z.append(Z[start:end])


        all_R = np.array(all_R)
        all_Z = np.array(all_Z)
        dihedrals = np.array(dihedrals)
        
        
        d = {}
        layers = pl_module.model.representation.n_interactions
        for l in range(layers):

            d[f"so3krates_layer_{l}"] = {}
            chi_out = pl_module.model.representation.so3krates_layer[l].record["chi_out"].detach().cpu().numpy()
            chi_in = pl_module.model.representation.so3krates_layer[l].record["chi_in"].detach().cpu().numpy()
            feature_out = pl_module.model.representation.so3krates_layer[l].record["features_out"].detach().cpu().numpy()
            feature_in = pl_module.model.representation.so3krates_layer[l].record["features_in"].detach().cpu().numpy()


            d[f"so3krates_layer_{l}"]["chi_in"] = chi_in
            d[f"so3krates_layer_{l}"]["chi_out"] = chi_out
            d[f"so3krates_layer_{l}"]["feature_in"] = feature_in
            d[f"so3krates_layer_{l}"]["feature_out"] = feature_out


            geo_att = pl_module.model.representation.so3krates_layer[l].geometry_block.attention_fn.record["alpha"].detach().cpu().numpy()
            fea_att = pl_module.model.representation.so3krates_layer[l].feature_block.attention_fn.record["alpha"].detach().cpu().numpy()
            #geo_att = pl_module.model.representation.so3krates_layer[l].geometry_block.attention_fn.record
            #fea_att = pl_module.model.representation.so3krates_layer[0].feature_block.attention_fn.record

            d[f"so3krates_layer_{l}"][f"geometric_head_alpha"] = geo_att
            d[f"so3krates_layer_{l}"][f"feature_head_alpha"] = fea_att


        prediction.update({"dihedrals":dihedrals})
        prediction.update({"R":all_R})
        prediction.update({"Z":all_Z})
        prediction.update({"so3krates":d})

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


class So3kratesCallback(Callback):
    """this is just a simple callback to collect the sphc coordinates
    right now the structure is not ideal, and might be changed later """

    def __init__(self):
        super().__init__()

    def on_fit_end(self, trainer, pl_module: AtomisticTask) -> None:

        d = {}
        layers = pl_module.model.representation.n_interactions
        for l in range(layers):

            d[f"so3krates_layer_{l}"] = {}
            chi_out = pl_module.model.representation.so3krates_layer[l].record["chi_out"].detach().cpu().numpy()
            chi_in = pl_module.model.representation.so3krates_layer[l].record["chi_in"].detach().cpu().numpy()
            
            d[f"so3krates_layer_{l}"]["chi_in"] = chi_in
            d[f"so3krates_layer_{l}"]["chi_out"] = chi_out

            geo_heads = pl_module.model.representation.so3krates_layer[l].geometry_block.attention_fn.record.keys()
            fea_heads = pl_module.model.representation.so3krates_layer[l].feature_block.attention_fn.record.keys()
            geo_att = pl_module.model.representation.so3krates_layer[l].geometry_block.attention_fn.record
            fea_att = pl_module.model.representation.so3krates_layer[0].feature_block.attention_fn.record

            for g in geo_heads:
                d[f"so3krates_layer_{l}"][f"geo_head_{g}"] = {}
                val = geo_att[g].detach().cpu().numpy()
                d[f"so3krates_layer_{l}"][f"geo_head_{g}"] = val


            for f in fea_heads:
                d[f"so3krates_layer_{l}"][f"fea_head_{f}"] = fea_att[f].detach().cpu().numpy()


            np.savez(os.path.join(trainer.log_dir,"so3krates_records.npz"), **d)

        print("Saving sow equivalent for attention weights and chi")
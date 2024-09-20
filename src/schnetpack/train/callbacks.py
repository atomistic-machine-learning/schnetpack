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
import pandas as pd
from tqdm import tqdm

__all__ = ["ModelCheckpoint", "PredictionWriter", "ExponentialMovingAverage"]



# class ENWriter(Callback):

#     def __init__(self, output_dir: str, write_interval: str):
#         """
#         Args:
#             output_dir: output directory for prediction files
#             write_interval: can be one of ["batch", "epoch", "batch_and_epoch"]
#         """
#         super().__init__()
#         self.output_dir = output_dir
#         self.write_interval = write_interval
#         os.makedirs(output_dir, exist_ok=True)

#     def on_train_epoch_end(
#         self,
#         trainer,
#         pl_module: AtomisticTask,
#         predictions: List[Any],
#         batch_indices: List[Any],
#     ):
#         bdir = os.path.join(self.output_dir, str(dataloader_idx))
#         os.makedirs(bdir, exist_ok=True)

#         # get the output of the model
#         output = outputs[0]
#         # get the input of the model
#         inputs = batch

class CustomLRSchedulerCallback(Callback):
    def __init__(self, threshold: float, target_lr: float):
        """
        Args:
            threshold (float): The validation loss threshold below which the learning rate is set.
            target_lr (float): The new learning rate to set if the validation loss falls below the threshold.
        """
        super().__init__()
        self.threshold = threshold
        self.target_lr = target_lr
        # we only want to to do this once
        self.counter = 0 

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Get the last validation loss from the logger (or from trainer if no logger is used)
        val_loss = trainer.callback_metrics.get("val_loss")

        # Check if validation loss is below the threshold
        if val_loss is not None and val_loss < self.threshold and self.counter == 0:
            # Update learning rate to the target value
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.target_lr

            print(f"Validation loss fell below {self.threshold}. Learning rate set to {self.target_lr}.")
            self.counter += 1

import numpy as np
import pandas as pd
from collections import defaultdict



import h5py
import numpy as np

class EmbeddingWriterV3(Callback):

    def __init__(self, file_name: str, writer_interval: int, max_examples_per_atom_type: int = 25000):
        self.file_name = file_name
        self.writer_interval = writer_interval
        self.max_examples_per_atom_type = max_examples_per_atom_type
        
        # Initialize atom counts
        self.atom_count = {int(k): 0 for k in range(1, 100)}
        
        # Create or load the HDF5 file and initialize groups if not present
        with h5py.File(self.file_name, 'w') as f:
            for atom_type in range(1, 100):
                if str(atom_type) not in f:
                    f.create_group(str(atom_type))
        
        self.batch_counter = 0
        self.data_accumulator = []

    def on_predict_batch_end(self, trainer, pl_module: AtomisticTask, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # Extract necessary data from the batch
        atomic_numbers = batch["_atomic_numbers"].detach().cpu().numpy().reshape(-1)
        embeddings = {
            "initial_nuclear_embedding": batch["initial_nuclear_embedding"].detach().cpu().numpy(),
            "scalar representation MP_1": batch["scalar representation MP_1"].detach().cpu().numpy(),
            "scalar representation MP_2": batch["scalar representation MP_2"].detach().cpu().numpy(),
            "scalar representation MP_3": batch["scalar representation MP_3"].detach().cpu().numpy()
        }

        # Collect the data for each atom type
        for atom_type in tqdm(np.unique(atomic_numbers)):
            atom_indices = np.where(atomic_numbers == atom_type)[0]
            atom_count = len(atom_indices)

            # If we already have 25k examples for this atom type, skip further writing
            if self.atom_count[atom_type] >= self.max_examples_per_atom_type:
                continue

            # Calculate how many more examples we can write for this atom type
            remaining_slots = self.max_examples_per_atom_type - self.atom_count[atom_type]
            selected_indices = atom_indices[:remaining_slots]  # Limit to remaining slots
            
            # Accumulate data for the selected indices
            self.data_accumulator.append((atom_type, {key: embeddings[key][selected_indices] for key in embeddings}))
            self.atom_count[atom_type] += len(selected_indices)

        # Write to HDF5 after the specified interval
        self.batch_counter += 1
        if self.batch_counter % self.writer_interval == 0:
            self._write_to_hdf5()

    def _write_to_hdf5(self):
        # Open the HDF5 file and write accumulated data
        with h5py.File(self.file_name, 'a') as f:
            for atom_type, data in self.data_accumulator:
                atom_group = f[str(atom_type)]

                # For each dataset (embedding/representation), append new data
                for key, values in data.items():
                    if key in atom_group:
                        # Append to existing dataset
                        dset = atom_group[key]
                        dset.resize(dset.shape[0] + values.shape[0], axis=0)
                        dset[-values.shape[0]:] = values
                    else:
                        # Create new dataset if it doesn't exist
                        maxshape = (None, values.shape[1])  # Set maxshape to None to allow appending
                        dset = atom_group.create_dataset(key, data=values, maxshape=maxshape, chunks=True)

        # Clear the data accumulator after writing
        self.data_accumulator = []



class EmbeddingWriterV2(Callback):

    def __init__(self, file_name: str, writer_interval: int):
        # Initialize the atomic count dictionary
        self.MP1_atom_count = defaultdict(int)
        
        # Prepare the column names
        self.cols = ["atomic_numbers"]
        for k in ["initial_nuclear_embedding", "scalar representation MP_1", "scalar representation MP_2", "scalar representation MP_3"]:
            self.cols.extend([f"{k}_Feature {i}" for i in range(256)])
        
        # Initialize an empty dataframe in memory
        self.data_accumulator = pd.DataFrame(columns=self.cols)
        
        # File details
        self.file_name = file_name
        self.writer_interval = writer_interval
        self.batch_counter = 0

    def on_predict_batch_end(self, trainer, pl_module: AtomisticTask, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # Extract the necessary keys
        keys = ["_atomic_numbers", "initial_nuclear_embedding", "scalar representation MP_1", "scalar representation MP_2", "scalar representation MP_3"]

        # Prepare data for the current batch
        batch_data = []
        for k in keys:
            if k == "_atomic_numbers":
                batch_data.append(batch[k].detach().cpu().numpy().reshape(-1, 1))
            else:
                batch_data.append(batch[k].detach().cpu().numpy())

        # Concatenate the data along the axis
        batch_data = np.concatenate(batch_data, axis=1)
        
        # Create a DataFrame from the concatenated data
        batch_df = pd.DataFrame(batch_data, columns=self.cols)
        atomic_numbers = batch_df["atomic_numbers"].values

        # Count atom types and filter data
        for atom_num, count in zip(*np.unique(atomic_numbers, return_counts=True)):
            if self.MP1_atom_count[atom_num] + count <= 2000:
                self.MP1_atom_count[atom_num] += count
            else:
                allowed_count = 2000 - self.MP1_atom_count[atom_num]
                if allowed_count > 0:
                    self.MP1_atom_count[atom_num] = 2000
                    batch_df = batch_df[batch_df["atomic_numbers"] == atom_num][:allowed_count]

        # Accumulate the data in memory
        self.data_accumulator = pd.concat([self.data_accumulator, batch_df], axis=0)

        # Write to CSV after the specified interval
        self.batch_counter += 1
        if self.batch_counter % self.writer_interval == 0:
            self._write_to_file()

    def _write_to_file(self):
        # Load the existing data, if any
        try:
            existing_data = pd.read_csv(self.file_name, index_col="atomic_numbers")
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=self.cols).set_index("atomic_numbers")

        # Concatenate the existing data with the accumulated data
        combined_data = pd.concat([existing_data, self.data_accumulator], axis=0)

        # Write the combined data to the CSV file
        combined_data.to_csv(self.file_name)

        # Clear the accumulator to free up memory
        self.data_accumulator = pd.DataFrame(columns=self.cols)



class EmbeddingWriter(Callback):

    def __init__(
            self,
            file_name:str,
            writer_interval: int):
        

        self.MP1_atom_count = {int(k) : 0 for k in range(1,100)}

        cols = ["atomic_numbers"]
        for k in ["initial_nuclear_embedding","scalar representation MP_1","scalar representation MP_2","scalar representation MP_3"]:
            c = [f"{k}_Feature {i}" for i in range(256)]
            cols.extend(c)

        for k in range(1,100):
            df = pd.DataFrame(columns=cols)
            df.set_index("atomic_numbers",inplace=True)
            df.to_csv(f"/home/elron/phd/projects/google/qmml/experiments/embedding_dfs/{k}_embeds.csv")

        self.file_name = file_name
        self.writer_interval = writer_interval
        self.writing_keys = [
                "scalar_representation", "initial_nuclear_embedding","_idx_i","_idx_j","_idx_m","_n_atoms","_atomic_numbers","charge"
                ]
        
        #["MP_1","MP_2","MP_3"]
        #f"scalar representation MP_{i+1}"
        # init the file 
        #with h5py.File(self.file_name, 'w') as f:
        #    for p in self.writing_keys :
                #f.create_group(p)
        #        g = f.create_group(p)
                #g.attrs["MP_1_atom_count"] = d
                #g.attrs["MP_1_atom_count"] = d
                #g.attrs["MP_1_atom_count"] = d
                # g.attrs["cutoff_shell"] = cutoff_shell
                # g.attrs["cell"] = np.array(cell)
                # g.attrs["n_replicas"] = n_replicas
                # g.attrs["system_temperature"] = system_temperature
                # g.attrs["model_id"] = model_id

    def on_predict_batch_end(self,
        trainer,
        pl_module: AtomisticTask,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0):
        
        # tmp create dataframe to store the data
        keys = ["_atomic_numbers","initial_nuclear_embedding","scalar representation MP_1","scalar representation MP_2","scalar representation MP_3"]
        cols = []


        data = []
        for k in keys:
            if k == "_atomic_numbers":
                data.append(batch[k].detach().cpu().numpy().reshape(-1,1))
                cols.append("atomic_numbers")
            else:
                data.append(batch[k].detach().cpu().numpy())
                c = [f"{k}_Feature {i}" for i in range(data[-1].shape[1])]
                cols.extend(c)


        data = np.concatenate(data,axis=1)
        data = pd.DataFrame(data,columns=cols)
        data.set_index("atomic_numbers",inplace=True)



        # count the atom types to later prevent exorbitant big file
        counts = data.index.value_counts().to_dict()
        for k in tqdm(counts.keys()):
            
            if self.MP1_atom_count[k] > 25000:
                continue
                #data.drop(data[data.index == k].index, inplace=True)

            else:
                df = pd.read_csv(f"/home/elron/phd/projects/google/qmml/experiments/embedding_dfs/{int(k)}_embeds.csv",index_col="atomic_numbers")
                df = pd.concat([df,data[data.index == k]],axis=0)
                df.to_csv(f"/home/elron/phd/projects/google/qmml/experiments/embedding_dfs/{int(k)}_embeds.csv")
                self.MP1_atom_count[k] += counts[k]


        #df = pd.read_csv(self.file_name,index_col="atomic_numbers")
        #df = pd.concat([df,data],axis=0)
        #df.to_csv(self.file_name)
#        if batch_idx % self.writer_interval == 0:
            #q,mu = (batch["scalar_representation"],batch["vector_representation"])
            #idx_i,idx_j,idx_m,n_atoms = (batch["_idx_i"],batch["_idx_j"],batch["_idx_m"],batch["_n_atoms"])
            #atomic_numbers = batch[properties.Z]
#            tag = "global step "+str(trainer.global_step)+"batch idx "+str(batch_idx)



#            with h5py.File(self.file_name, 'a') as f:
                
#                for key in self.writing_keys:
#                    f[key].create_dataset(tag, data=batch[key].detach().cpu().numpy(),compression="gzip")

                #f["scalar_representation"].create_dataset(tag, data=q.detach().cpu().numpy(),compression="gzip") 
                ##f["vector_representation"].create_dataset(tag, data=mu.detach().cpu().numpy(),compression="gzip")
                #f["idx_i"].create_dataset(tag, data=idx_i.detach().cpu().numpy(),compression="gzip")
                #f["idx_j"].create_dataset(tag, data=idx_j.detach().cpu().numpy(),compression="gzip")
                #f["idx_m"].create_dataset(tag, data=idx_m.detach().cpu().numpy(),compression="gzip")
                #f["n_atoms"].create_dataset(tag, data=n_atoms.detach().cpu().numpy(),compression="gzip")
                #f["atomic_numbers"].create_dataset(tag, data=atomic_numbers.detach().cpu().numpy(),compression="gzip")
                #f["charge"].create_dataset(tag, data=batch["charge"].detach().cpu().numpy(),compression="gzip")                

        
    

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

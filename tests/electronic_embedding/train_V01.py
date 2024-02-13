'''
# =================================================================
# INTENTION
# =================================================================

This is only a simple training script to run a schnet model
with either SchNet or PaiNN (not yet implemented) representation.

Right now one can train a SchNet model with charge and spin embedding
for carbene and ag3 ions. The datasets are the same as in 
https://doi.org/10.1038/s41467-021-27504-0 (SpookyNet)

The datasets are stored in the tests/electronic_embedding folder
The input file is a json file stored in tests/electronic_embedding/carbene.json
The usage is just intended for quick testing purposes. It is advisable to switch
to hydra configs once proven the charge and spin embedding works as intended.

# =================================================================
# USAGE
# =================================================================

local: python train_for_new_feature.py /test/carbene.json
hydra_ml_cluster: see related bash script train.sh stored in tests/electronic_embedding
'''


# ===========================================
# IMPORTS
# ===========================================
import os
import uuid
from datetime import datetime
import sys
import json
import wandb

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.train.lr_scheduler import ReduceLROnPlateau

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# ===========================================
# Function definitions
# ===========================================
def read_input_file(inputfile):
    with open(inputfile) as json_file:
        data = json.load(json_file)
        print('Argument file succesfully loaded')
        return data



# ===========================================
# Loading input args
# ===========================================
#inputfile = sys.argv[1]
inputfile = "/home/elron/phd/projects/schnetpack/tests/electronic_embedding/pain_carbene.json"
inputs_args = read_input_file(inputfile)
# for logger watch
watch = False

# dataset config specific
db_path = inputs_args["db_path"]
cutoff = inputs_args["cutoff"]
load_properties = inputs_args["load_properties"]
property_units = inputs_args["property_units"]


# trainer config specific
num_epoch = inputs_args["num_epoch"]
lr = inputs_args["lr"]
save_dir = inputs_args["save_dir"]
batch_size = inputs_args["batch_size"]
num_val = inputs_args["num_val"]
num_train = inputs_args["num_train"]
split_file = inputs_args["split_file"]

# representation config specific
n_atom_basis = inputs_args["n_atom_basis"]
n_rbf = inputs_args["n_rbf"]
n_interaction = inputs_args["n_interaction"]
cutoff_fn = inputs_args["cutoff_fn"]
representation = inputs_args["representation"]
activate_charge_spin_embedding = inputs_args["activate_charge_spin_embedding"]
cutoff_fn = spk.nn.CosineCutoff(cutoff)
pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)


# logger config specific
id = str(uuid.uuid1())
date = str(datetime.now().year) + "-" + str(datetime.now().month) + "-" + str(datetime.now().day)
wandb_id = date + "_" + id
wandb_name = inputs_args["wandb_name"]

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ===========================================
# data preparation
# ===========================================

prepared_dataset = spk.data.AtomsDataModule(
    datapath=db_path,
    batch_size=batch_size,
    num_train=num_train,
    num_val=num_val,
    split_file=split_file,
    transforms=[
        trn.MatScipyNeighborList(cutoff=cutoff),
        trn.SubtractCenterOfMass(),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    property_units=property_units,
    distance_unit='Ang',
    load_properties=load_properties
)


# ===========================================
# representation
# ===========================================
repr_dict = {
    "schnet":spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=n_interaction,
    radial_basis=radial_basis,
    cutoff_fn=cutoff_fn,
    activate_charge_spin_embedding=activate_charge_spin_embedding),
    'painn':spk.representation.PaiNN(
    n_atom_basis=n_atom_basis, n_interactions=n_interaction,
    radial_basis=radial_basis,
    cutoff_fn=cutoff_fn,activate_charge_spin_embedding=activate_charge_spin_embedding)
}

repr = repr_dict[representation]

# ===========================================
# output module
# ===========================================
pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='energy')
pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")
# ===========================================
# build model
# ===========================================
nnpot = spk.model.NeuralNetworkPotential(
    representation=repr,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy,pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy",add_mean=True,add_atomrefs=False)
    ]
)

output_energy = spk.task.ModelOutput(
    name='energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.05,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_forces = spk.task.ModelOutput(
    name="forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.95,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)


task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy,output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": lr},
    scheduler_cls=ReduceLROnPlateau,
    scheduler_args = {
        "mode" : "min",
        "factor":0.5,
        "patience":20,
        "threshold":1e-4,
        "threshold_mode":"rel",
        "cooldown":10,
        "min_lr":1e-6,
        "smoothing_factor":0.0},
    scheduler_monitor="val_loss",
        
    )


# wandb logger because tensorboard logger does work for now on my local machine
wandb.login()
logger = WandbLogger(project='google_dataset', config=inputs_args,
                       name=wandb_name, id=wandb_id, resume='allow')

if watch:
    logger.watch(task, log="all",log_freq=False)

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(save_dir, 'checkpoints'),
        save_top_k=1,
        monitor="val_loss"
    )
]
trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=save_dir,
    max_epochs=num_epoch,
    accelerator="cuda" 
)
trainer.fit(task, datamodule=prepared_dataset)
wandb.unwatch(task)
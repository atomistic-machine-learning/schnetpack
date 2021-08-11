import logging
import os

import pytorch_lightning
import torch.optim
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

import schnetpack as spk
from schnetpack.datasets import MD17

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# basic settings
model_dir = "ethanol_model"
os.makedirs(model_dir)

batch_size = 10
cutoff = 5.0

logging.info("Setup dataset & preprocessing")

# transforms are applied to each example as a preprocessing step before batching and
# passing it to the network
transforms = [
    spk.transform.RemoveOffsets(
        property=MD17.energy, remove_mean=True, remove_atomrefs=False
    ),
    spk.transform.ASENeighborList(cutoff=cutoff),
    spk.transform.CastTo32(),
]

# the pytorch lightning datamodule handles downloading, partitioning and loading of data
dataset = MD17(
    "../data/ethanol.db",
    molecule="ethanol",
    batch_size=batch_size,
    num_train=950,
    num_val=50,
    distance_unit="Ang",
    property_units={MD17.energy: "eV", MD17.forces: "eV/Ang"},
    transforms=transforms,
)


logging.info("Build model")

# Each SchNetPack model consists of the representation (e.g. SchNet or PainNN)
# and a list of output modules
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
representation = spk.representation.SchNet(
    n_atom_basis=64,
    n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)
output_modules = [
    spk.atomistic.Atomwise(
        output=MD17.energy,
        n_in=representation.n_atom_basis,
    ),
    spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces),
]

# To put things together, we define model outputs including corresponding loss functions
# and metrics to log.
outputs = [
    spk.atomistic.ModelOutput(
        name=MD17.energy,
        loss_fn=torchmetrics.regression.MeanSquaredError(),
        loss_weight=0.05,
        metrics={
            "mse": torchmetrics.regression.MeanSquaredError(),
            "mae": torchmetrics.regression.MeanAbsoluteError(),
        },
    ),
    spk.atomistic.ModelOutput(
        name=MD17.forces,
        loss_fn=torchmetrics.regression.MeanSquaredError(),
        loss_weight=0.95,
        metrics={
            "mse": torchmetrics.regression.MeanSquaredError(),
            "mae": torchmetrics.regression.MeanAbsoluteError(),
        },
    ),
]

# put the model together
model = spk.atomistic.AtomisticModel(
    datamodule=dataset,
    representation=representation,
    output_modules=output_modules,
    outputs=outputs,
    optimizer_cls=torch.optim.Adam,
    optimizer_args={"lr": 5e-4},
)


# callbacks for PyTroch Lightning Trainer
logging.info("Setup trainer")
callbacks = [
    spk.train.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath="checkpoints",
        filename="{epoch:02d}",
        inference_path="best_inference_model",
        save_as_torch_script=False,
    ),
    pytorch_lightning.callbacks.EarlyStopping(
        monitor="val_loss", patience=150, mode="min", min_delta=0.0
    ),
    pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
]

logger = TensorBoardLogger("tensorboard/")
trainer = pytorch_lightning.Trainer(callbacks=callbacks, logger=logger, gpus=0)

logging.info("Start training")
trainer.fit(model=model, datamodule=dataset)

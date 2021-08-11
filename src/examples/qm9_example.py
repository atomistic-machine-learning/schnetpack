import logging
import os

import pytorch_lightning
import torch.optim
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

import schnetpack as spk
from schnetpack.datasets import QM9

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# basic settings
model_dir = "qm9_model"
os.makedirs(model_dir)

batch_size = 64
cutoff = 5.0

logging.info("Setup dataset & preprocessing")

# transforms are applied to each example as a preprocessing step before batching and
# passing it to the network
transforms = [
    spk.transform.RemoveOffsets(
        property=QM9.U0, remove_mean=True, remove_atomrefs=True
    ),
    spk.transform.ASENeighborList(cutoff=cutoff),
    spk.transform.CastTo32(),
]

# the pytorch lightning datamodule handles downloading, partitioning and loading of data
dataset = QM9(
    "./data/qm9.db",
    batch_size=batch_size,
    num_train=10000,
    num_val=1000,
    distance_unit="Ang",
    property_units={QM9.U0: "eV"},
    transforms=transforms,
    remove_uncharacterized=False,
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
        output=QM9.U0,
        n_in=representation.n_atom_basis,
    )
]

# To put thing togethre, we define model outputs including corresponding loss functions
# and metrics to log.
outputs = [
    spk.atomistic.ModelOutput(
        name=QM9.U0,
        loss_fn=torchmetrics.regression.MeanSquaredError(),
        loss_weight=1.0,
        metrics={
            "mse": torchmetrics.regression.MeanSquaredError(),
            "mae": torchmetrics.regression.MeanAbsoluteError(),
        },
    )
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

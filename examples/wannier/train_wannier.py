import os

import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from schnetpack.atomistic.wannier import WannierCenter
from schnetpack.wannier.utils import instantiate_class

## Model training
max_epoch_val = 3  # setting 3 for demo. Use at least 1500
cutoff = 5.0
batch_size = 10
val_batch_size = 20
num_workers = 1
lr = 1e-4

##Setting parameters in schnetpack
n_atom_basis = 30  # number of features to describe atomic environments
n_interactions = 3  # number of interaction blocks (number of layers)
n_rbf = 20  # Total Number of Gaussian functions

## Loss functions and checkpoints settings
loss_fn = torch.nn.MSELoss()  # Using Mean Squared Error as loss function
loss_weight = 1.0
optimizer_cls = torch.optim.AdamW  # Using Adams optimizer
monitor = "val_loss"  # Criteria for checkpoint of saving best model 'train_loss'

hyperparameter_dict = {
    "max_epoch_val": max_epoch_val,
    "cutoff_radius": cutoff,
    "batch_size": batch_size,
    "val_batch_size": val_batch_size,
    "num_workers": num_workers,
    "lr": lr,
    "n_atom_basis": n_atom_basis,
    "n_layers": n_interactions,
    "n_rad_basis": n_rbf,
    "loss": str(loss_fn),
    "loss_weight": loss_weight,
    "optimizer": str(optimizer_cls),
    "monitor": monitor,
}

qm9tut = "./qm9tut"
if not os.path.exists(qm9tut):
    os.makedirs(qm9tut)

processed = "/Users/mjwen.admin/Desktop/sharing_with_Dr_wen/processed/"
db_path = processed + "wannier_dataset.db"
split_path = processed + "split.npz"

custom_data = spk.data.AtomsDataModule(
    db_path,
    batch_size=batch_size,
    val_batch_size=val_batch_size,
    split_file=split_path,
    distance_unit="Ang",
    property_units={"wan": "Ang"},
    transforms=[trn.ASENeighborList(cutoff=cutoff), trn.CastTo32()],
    num_workers=num_workers,
    pin_memory=False,  # set to false, when not using a GPU, set to True when using a GPU
)
# print(custom_data.val_idx)

# print(custom_data.val_idx)


if __name__ == "__main__":
    pairwise_distance = (
        spk.atomistic.PairwiseDistances()
    )  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    schnet = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    print("All okay")
    pred_wan = WannierCenter(
        n_in=n_atom_basis,
        dipole_key="wan",
    )

    # print(pred_wan.return_charges)
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_wan],
        postprocessors=[trn.CastTo64()],
    )

    output_wan = spk.task.ModelOutput(
        name="wan",
        loss_fn=loss_fn,
        loss_weight=loss_weight,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_wan],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={
            "mode": "min",
            "factor": 0.1,
            "patience": 15,
            "threshold": 1e-5,
            "verbose": True,
        },
        scheduler_monitor="val_loss",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
        stopping_threshold=1e-06,
        min_delta=0.0,
    )

    # pl.LightningModule.save_hyperparameters(pl, ignore=["model"])

    callbacks = [
        spk.train.ModelCheckpoint(
            mode="min",
            model_path=os.path.join(qm9tut, "best_inference_model"),
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
        ),
        early_stopping_callback,
    ]
    # logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
    print(callbacks[0].model_path)

    wandb_logger = WandbLogger(
        project="demo",
        save_dir=qm9tut,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=callbacks,
        logger=wandb_logger,
        # log_every_n_steps=1,
        default_root_dir=qm9tut,
        max_epochs=max_epoch_val,  # for testing, we restrict the number of epochs
    )
    print(trainer.default_root_dir)

    trainer.fit(task, datamodule=custom_data)
    print("After fitting ", trainer.default_root_dir)

    best_model_path = callbacks[0].best_model_path
    print("best_model_path", best_model_path)

    best_train_loss = (trainer.callback_metrics["train_loss"]).item()
    best_val_loss = (trainer.callback_metrics["val_loss"]).item()

    # mjwen
    trainer.test(ckpt_path=best_model_path, datamodule=custom_data)

    print(f"Best Training Loss: {best_train_loss}")
    print(f"Best Validation Loss: {best_val_loss}")

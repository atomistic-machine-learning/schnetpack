import logging
from torch.optim import Adam
import os
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.datasets import QM9
from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError
from schnetpack.train import build_mse_loss


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# basic settings
model_dir = "qm9_model"
os.makedirs(model_dir)
properties = [QM9.U0]

# data preparation
logging.info("get dataset")
dataset = QM9("data/qm9.db", load_only=[QM9.U0])
train, val, test = spk.train_test_split(
    dataset, 1000, 100, os.path.join(model_dir, "split.npz")
)
train_loader = spk.AtomsLoader(train, batch_size=64, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=64)

# statistics
atomrefs = dataset.get_atomref(properties)
means, stddevs = train_loader.get_statistics(
    properties, divide_by_atoms=True, single_atom_ref=atomrefs
)

# model build
logging.info("build model")
representation = spk.SchNet(n_interactions=6)
output_modules = [
    spk.atomistic.Atomwise(
        n_in=representation.n_atom_basis,
        property=QM9.U0,
        mean=means[QM9.U0],
        stddev=stddevs[QM9.U0],
        atomref=atomrefs[QM9.U0],
    )
]
model = schnetpack.AtomisticModel(representation, output_modules)

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError(p, p) for p in properties]
hooks = [CSVHook(log_path=model_dir, metrics=metrics), ReduceLROnPlateauHook(optimizer)]

# trainer
loss = build_mse_loss(properties)
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# run training
logging.info("training")
trainer.train(device="cpu")

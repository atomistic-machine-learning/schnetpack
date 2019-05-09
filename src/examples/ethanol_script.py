import os
import logging
from shutil import rmtree
from torch.optim import Adam
from schnetpack.atomistic import AtomisticModel
from schnetpack.output_modules import Atomwise
from schnetpack.data import AtomsData, AtomsLoader, train_test_split
from schnetpack.representation import SchNet
from schnetpack.train import Trainer, TensorboardHook, CSVHook, ReduceLROnPlateauHook
from schnetpack.metrics import MeanAbsoluteError
from schnetpack.utils import loss_fn


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# basic settings
db_path = "data/md17/ethanol.db"  # relative path to the database file
model_dir = "ethanol_model"  # directory that will be created for storing model
properties = ["energy", "forces"]  # properties used for training
num_train, num_val = 1000, 100  # number of training and validating samples
batch_size = 64  # batch size used in training
device = "cpu"  # device used, choose between 'cpu' & 'gpu'

# create folders
logging.info("datapath: {}".format(db_path))
if os.path.exists(model_dir):
    rmtree(model_dir)
os.makedirs(model_dir)

# data preparation
logging.info("get dataset")
dataset = AtomsData(db_path, required_properties=properties)
train, val, test = train_test_split(
    data=dataset,
    num_train=num_train,
    num_val=num_val,
    split_file=os.path.join(model_dir, "split.npz"),
)
train_loader = AtomsLoader(train, batch_size=batch_size)
val_loader = AtomsLoader(val, batch_size=batch_size)
test_loader = AtomsLoader(test, batch_size=batch_size)

# get statistics
atomrefs = dataset.get_atomrefs(properties)
per_atom = dict(energy=True, forces=False)
means, stddevs = train_loader.get_statistics(
    properties, atomrefs=atomrefs, per_atom=per_atom
)

# model build
logging.info("build model")
representation = SchNet(n_interactions=6)
output_modules = [
    Atomwise(
        property="energy",
        derivative="forces",
        mean=means["energy"],
        stddev=stddevs["energy"],
        negative_dr=True,
    )
]
model = AtomisticModel(representation, output_modules)

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError(p, p) for p in properties]
logging_hooks = [
    TensorboardHook(log_path=model_dir, metrics=metrics),
    CSVHook(log_path=model_dir, metrics=metrics),
]
scheduling_hooks = [ReduceLROnPlateauHook(patience=25, window_length=3, factor=0.8)]
hooks = logging_hooks + scheduling_hooks

# trainer
loss = loss_fn(properties)
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=Adam(params=model.parameters(), lr=1e-4),
    train_loader=train_loader,
    validation_loader=val_loader,
)

# run training
logging.info("training")
trainer.train(device=device, n_epochs=10000)

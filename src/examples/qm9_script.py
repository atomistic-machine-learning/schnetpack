import logging
from torch.optim import Adam
import os
from shutil import rmtree
from schnetpack.atomistic import AtomisticModel
from schnetpack.output_modules import Atomwise
from schnetpack.datasets import QM9
from schnetpack.data import AtomsLoader, train_test_split
from schnetpack.representation import SchNet
from schnetpack.train import Trainer, TensorboardHook, CSVHook, ReduceLROnPlateauHook
from schnetpack.metrics import MeanAbsoluteError
from schnetpack.utils import loss_fn


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# basic settings
db_path = "data/qm9.db"
model_dir = "qm9_model"
num_train, num_val = [1000, 100]
batch_size = 64
device = "cpu"

# create folders
logging.info("datapath: {}".format(db_path))
if os.path.exists(model_dir):
    rmtree(model_dir)
os.makedirs(model_dir)

# data preparation
logging.info("get dataset")
properties = [QM9.U0, QM9.homo, QM9.lumo]
dataset = QM9(db_path)
train, val, test = train_test_split(
    data=dataset,
    num_train=num_train,
    num_val=num_val,
    split_file=os.path.join(model_dir, "split.npz"),
)
train_loader = AtomsLoader(train, batch_size=batch_size)
val_loader = AtomsLoader(val, batch_size=batch_size)
test_loader = AtomsLoader(test, batch_size=batch_size)

# statistics
atomrefs = dataset.get_atomrefs(properties)
means, stddevs = train_loader.get_statistics(
    properties, per_atom=True, atomrefs=atomrefs
)

# model build
logging.info("build model")
representation = SchNet(n_interactions=6)
output_modules = [
    Atomwise(property=p, mean=means[p], stddev=stddevs[p], atomref=atomrefs[p])
    for p in properties
]
model = AtomisticModel(representation, output_modules)

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError(p, p) for p in properties]
logging_hooks = [
    TensorboardHook(log_path=model_dir, metrics=metrics),
    CSVHook(log_path=model_dir, metrics=metrics),
]
scheduleing_hooks = [ReduceLROnPlateauHook(patience=25, window_length=3, factor=0.8)]
hooks = logging_hooks + scheduleing_hooks

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
trainer.train(device=device)

import torch
from torch.optim import Adam
import os
from shutil import rmtree
from schnetpack.atomistic import PropertyModel, NewAtomisticModel, Atomwise
from schnetpack.datasets import QM9
from schnetpack.data import AtomsLoader, train_test_split
from schnetpack.representation import SchNet
from schnetpack.train import Trainer, TensorboardHook, CSVHook, ReduceLROnPlateauHook
from schnetpack.metrics import MeanAbsoluteError


def loss_fn(properties):
    """
    Build the loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties

    Returns:
        loss function

    """

    def loss_fn(batch, result):
        loss = 0.0
        for prop in properties:
            diff = batch[prop] - result[prop]
            diff = diff ** 2
            err_sq = torch.mean(diff)
            loss += err_sq
        return loss
    return loss_fn


# basic settings
batch_size = 50
data_dir = 'data'
model_dir = 'training'
os.makedirs(data_dir, exist_ok=True)
if os.path.exists(model_dir):
    rmtree(model_dir)
os.makedirs(model_dir)

# data preparation
properties = [QM9.U0]#, QM9.U0, QM9.homo, QM9.lumo]
dataset = QM9(os.path.join(data_dir, 'qm9.db'))
train, val, test = train_test_split(data=dataset, num_train=1000, num_val=100,
                                    split_file='training/split.npz')
train_loader = AtomsLoader(train, batch_size=batch_size)
val_loader = AtomsLoader(val, batch_size=batch_size)
test_loader = AtomsLoader(test, batch_size=batch_size)
atomrefs = {p: dataset.get_atomref(p) for p in properties}
mean, stddev = train_loader.get_statistics(properties, atomrefs=atomrefs)

# model build
representation = SchNet(n_interactions=6)
output_modules = [Atomwise(property=p, mean=mean[0], stddev=stddev[0]) for p in
                  properties]
property_model = PropertyModel(output_modules=output_modules)
model = NewAtomisticModel(representation, property_model)

# hooks
metrics = [MeanAbsoluteError(p, p) for p in properties]
logging_hooks = [TensorboardHook(log_path=model_dir, metrics=metrics),
                 CSVHook(log_path=model_dir, metrics=metrics)]
scheduleing_hooks = [ReduceLROnPlateauHook(patience=25, window_length=3)]
hooks = logging_hooks + scheduleing_hooks

# trainer
loss = loss_fn(properties)
trainer = Trainer(model_dir, model=model, hooks=hooks, loss_fn=loss,
                  optimizer=Adam(params=model.parameters()), train_loader=train_loader,
                  validation_loader=val_loader)

# run training
trainer.train(device='cpu')

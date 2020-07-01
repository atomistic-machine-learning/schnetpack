import schnetpack.atomistic.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack.representation as rep
from schnetpack.datasets import *
import schnetpack as spk
from shutil import rmtree

# load qm9 dataset and download if necessary
print("loading data...")
data = QM9("data/schnetpack/qm9.db")

# split in train and val
print("creating splits...")
train, val, test = spk.data.train_test_split(data, 1000, 100)
loader = spk.data.AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
print("creating model...")
# representation
reps = rep.PhysNet(activation=spk.nn.shifted_softplus,
                   distance_expansion=spk.nn.ExponentialGaussianFunctions(32))
# output module as modular wrapper and atomwise layer
corrections = [spk.atomistic.ElectrostaticEnergy(cuton=0., cutoff=10.)]
output_layer = schnetpack.atomistic.AtomwiseCorrected(
    n_in=reps.n_atom_basis, corrections=corrections, property=QM9.U0
)
out2 = schnetpack.atomistic.Atomwise(
    n_in=reps.n_atom_basis,
    atomref=data.atomref,
    property=QM9.U0,
    activation=spk.nn.shifted_softplus,
)

# atomistic model
model = schnetpack.atomistic.AtomisticModel(reps, out2)

# create trainer
print("setting up trainer...")
modeldir = "modeldir"
rmtree(modeldir)
opt = Adam(model.parameters(), lr=1e-4)
loss = lambda b, p: F.mse_loss(p[QM9.U0], b[QM9.U0])
trainer = spk.train.Trainer(modeldir, model, loss, opt, loader, val_loader)

# start training
print("training...")
trainer.train(torch.device("cpu"))

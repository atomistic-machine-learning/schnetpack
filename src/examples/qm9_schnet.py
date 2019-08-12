import schnetpack.atomistic.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack as spk
import schnetpack.representation as rep
from schnetpack.datasets import *

# load qm9 dataset and download if necessary
data = QM9("qm9.db")

# split in train and val
train, val, test = data.create_splits(100000, 10000)
loader = spk.data.AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
reps = rep.SchNet()
output = schnetpack.atomistic.Atomwise(n_in=reps.n_atom_basis)
model = schnetpack.atomistic.AtomisticModel(reps, output)

# create trainer
opt = Adam(model.parameters(), lr=1e-4)
loss = lambda b, p: F.mse_loss(p["y"], b[QM9.U0])
trainer = spk.train.Trainer("output/", model, loss, opt, loader, val_loader)

# start training
trainer.train(torch.device("cpu"))

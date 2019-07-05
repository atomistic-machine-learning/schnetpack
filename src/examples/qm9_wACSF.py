import schnetpack.atomistic.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack as spk
import schnetpack.representation as rep
from schnetpack.datasets import *

# load qm9 dataset and download if necessary
data = QM9("qm9.db", collect_triples=True)

# split in train and val
train, val, test = data.create_splits(100000, 10000)
loader = spk.data.AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
reps = rep.BehlerSFBlock()
output = schnetpack.atomistic.ElementalAtomwise(reps.n_symfuncs)
model = schnetpack.atomistic.AtomisticModel(reps, output)

# filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())

# create trainer
opt = Adam(trainable_params, lr=1e-4)
loss = lambda b, p: F.mse_loss(p["y"], b[QM9.U0])
trainer = spk.train.Trainer("wacsf/", model, loss, opt, loader, val_loader)

# start training
trainer.train(torch.device("cpu"))

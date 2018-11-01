import torch
from torch.optim import Adam

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *
from  schnetpack.loss_functions import MSELoss


# load qm9 dataset and download if necessary
data = QM9("qm9/", properties=[QM9.U0], collect_triples=True)

# split in train and val
train, val, test = data.create_splits(100000, 10000)
loader = spk.data.AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
reps = rep.BehlerSFBlock()
output = atm.ElementalAtomwise(reps.n_symfuncs)
model = atm.AtomisticModel(reps, output)

# create trainer
loss = MSELoss(input_key=QM9.U0, target_key='y')
trainer = spk.train.Trainer("output/", model, loss,
                            Adam, optimizer_params=dict(lr=1e-4))

# start training
trainer.train(torch.device("cpu"), loader, val_loader)

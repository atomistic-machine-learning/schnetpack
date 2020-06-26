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
n_atom_basis = 128
n_interactions = 6
n_gaussians = 32

# embedding
embedding = spk.nn.Embedding(n_features=n_atom_basis)

# distance expansion
distance_expansion = spk.nn.GaussianSmearing(n_gaussians=n_gaussians)

# interaction blocks
interactions = [
    spk.representation.PhysNetInteraction(
        n_features=n_atom_basis, n_gaussians=n_gaussians,
        activation=spk.nn.shifted_softplus
    ) for _ in range(n_interactions)
]

# post interaction network
post_interactions = None

# interaction aggregation mode
interaction_aggregation = spk.representation.InteractionAggregation(mode="sum")

# build representation
reps = rep.AtomisticRepresentation(
    embedding=embedding,
    distance_expansion=distance_expansion,
    interactions=interactions,
    post_interactions=post_interactions,
    interaction_aggregation=interaction_aggregation,
)

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

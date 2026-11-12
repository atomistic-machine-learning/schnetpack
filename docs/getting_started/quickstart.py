# %% [markdown]
# # Quickstart
#
# This page trains a small force field in Python, end to end, in about a minute on a
# CPU: build a dataset, train a PaiNN model on energies and forces, and use it as an
# ASE calculator.
#
# To stay that fast, the data is a toy: a short molecular dynamics run of ethanol
# with ASE's EMT calculator, a cheap empirical potential, in place of quantum
# chemistry. The [tutorials](../tutorials/data.py) go through each step in detail.

# %%
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from ase import units
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.lightning import AtomisticTask, AtomsDataModule

# everything this page writes goes here; use a directory of your own instead
workdir = tempfile.mkdtemp()

# %% [markdown]
# ## Data
#
# SchNetPack keeps a dataset in an ASE database. We fill one with 500 frames of an
# ethanol trajectory at 500 K and store the EMT energy and forces of every frame.

# %%
ethanol = molecule("CH3CH2OH")
ethanol.calc = EMT()
BFGS(ethanol, logfile=None).run(fmax=0.01)  # start from the EMT minimum

rng = np.random.default_rng(0)
atoms = ethanol.copy()
atoms.calc = EMT()
MaxwellBoltzmannDistribution(atoms, temperature_K=500, rng=rng)
md = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=500,
    friction=0.01 / units.fs,
    rng=rng,
)

atoms_list, property_list = [], []
for _ in range(500):
    md.run(10)  # 5 fs between frames
    atoms_list.append(atoms.copy())
    property_list.append(
        {
            "energy": np.array([atoms.get_potential_energy()]),
            "forces": atoms.get_forces(),
        }
    )

datapath = os.path.join(workdir, "ethanol_emt.db")
db = ASEAtomsData.create(
    datapath,
    distance_unit="Ang",
    property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
)
db.add_systems(property_list, atoms_list)

# %% [markdown]
# A dataset reads the structures from the database and passes each one through its
# transforms, here the neighbor list within a cutoff of 5 Å. The batch transforms
# act on each batch: they remove the mean energy, so the network learns the
# deviation from it, and cast to single precision.
#
# The data module splits the dataset into training, validation and test sets,
# computes the mean energy on the training set and builds a loader for each set.

# %%
cutoff = 5.0

dataset = ASEAtomsData(datapath, transforms=[trn.MatScipyNeighborList(cutoff=cutoff)])
datamodule = AtomsDataModule(
    dataset,
    batch_transforms=[trn.RemoveOffsets("energy", remove_mean=True), trn.CastTo32()],
    batch_size=32,
    num_train=400,
    num_val=50,
    split_file=os.path.join(workdir, "split.npz"),
)

# %% [markdown]
# ## Model
#
# A model is a pipeline of modules that read from and write to a dictionary of
# tensors. `PairwiseDistances` computes the distance vectors of the neighbor pairs,
# PaiNN turns them into a feature vector per atom, `Atomwise` sums an energy
# contribution per atom, and `Forces` differentiates the energy with respect to
# the positions. The postprocessors add the mean energy back and return double
# precision.

# %%
n_atom_basis = 32

model = spk.model.NeuralNetworkPotential(
    representation=spk.model.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=3,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    ),
    input_modules=[spk.model.PairwiseDistances()],
    output_modules=[
        spk.model.Atomwise(n_in=n_atom_basis, output_key="energy"),
        spk.model.Forces(energy_key="energy", force_key="forces"),
    ],
    postprocessors=[trn.CastTo64(), trn.AddOffsets("energy", add_mean=True)],
)

# %% [markdown]
# ## Training
#
# The task names what the model is trained on: a mean squared error on energy and
# forces, weighted towards the forces. It is a PyTorch Lightning module, so the
# Lightning `Trainer` runs the training.

# %%
outputs = [
    spk.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    ),
    spk.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    ),
]
task = AtomisticTask(
    model=model,
    outputs=outputs,
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-3},
)

logger = pl.loggers.CSVLogger(workdir)
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    # the forces are gradients, so validation and testing need autograd
    inference_mode=False,
    logger=logger,
    default_root_dir=workdir,
    enable_progress_bar=False,
    enable_model_summary=False,
)
pl.seed_everything(0)
trainer.fit(task, datamodule=datamodule)

# %% [markdown]
# The CSV logger wrote the validation metrics of every epoch:

# %%
import csv

with open(os.path.join(logger.log_dir, "metrics.csv")) as f:
    val = [row for row in csv.DictReader(f) if row["val_forces_MAE"]]

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(
    [int(row["epoch"]) for row in val],
    [float(row["val_forces_MAE"]) for row in val],
)
ax.set_xlabel("epoch")
ax.set_ylabel("force MAE (eV/Å)")
ax.set_yscale("log")
plt.show()

# %% [markdown]
# On the test set, which the training never saw:

# %%
trainer.test(task, datamodule=datamodule, verbose=False)
print(
    {k: round(float(v), 4) for k, v in trainer.callback_metrics.items() if "test" in k}
)

# %% [markdown]
# ## Using the model
#
# `SpkCalculator` wraps the model as an ASE calculator. It builds the neighbor list
# for every structure it is given and converts the units: the model was trained in
# eV, so we say so. Here it relaxes a distorted ethanol, and EMT checks the result.

# %%
calculator = spk.interfaces.SpkCalculator(
    model=task.model,
    neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff),
    energy_unit="eV",
    position_unit="Ang",
)

atoms = ethanol.copy()
atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
atoms.calc = calculator
BFGS(atoms, logfile=None).run(fmax=0.05, steps=200)

reference = atoms.copy()
reference.calc = EMT()
print(f"model energy: {atoms.get_potential_energy():.3f} eV")
print(f"EMT energy:   {reference.get_potential_energy():.3f} eV")
print(
    f"largest EMT force at the relaxed structure: "
    f"{np.abs(reference.get_forces()).max():.3f} eV/Å"
)

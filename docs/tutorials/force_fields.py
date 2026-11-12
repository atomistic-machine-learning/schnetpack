# %% [markdown]
# # Training a model on forces and energies
#
# This tutorial trains a neural network potential on energies and forces and runs
# molecular dynamics (MD) with it. It goes beyond the
# [quickstart](../getting_started/quickstart.py) in four points: why forces help,
# how the loss weighs energies against forces, how to keep the best model during
# training, and how to check the model in a simulation.
#
# The data is a stand-in so that the page runs in a few minutes on a CPU: an MD run
# of acetic acid with ASE's EMT calculator, a cheap empirical potential. The last
# section shows the same setup on the rMD17 benchmark, which has quantum-chemical
# energies and forces.

# %%
import csv
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
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEDBStorage, AtomsDataset
from schnetpack.lightning import AtomisticTask, AtomsDataModule, ModelCheckpoint

pl.seed_everything(0)
rng = np.random.default_rng(0)

# everything this page writes goes here; use a directory of your own instead
workdir = tempfile.mkdtemp()

# %% [markdown]
# ## Why train on forces
#
# The force on atom $i$ is the negative gradient of the potential energy $E$ with
# respect to its position $\mathbf{R}_i$:
#
# $$
# \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{R}_i}.
# $$
#
# This has two consequences for a model.
#
# - **More labels per structure.** A structure of $N$ atoms has one energy but $3N$
#   force components. Quantum chemistry codes compute the forces at little extra
#   cost, so every reference calculation teaches the model much more.
# - **Energy conservation.** Forces taken as the gradient of a predicted energy
#   form a conservative force field. An MD run with them conserves the total
#   energy, as a physical system does. A model that predicts forces directly does
#   not guarantee this.
#
# SchNetPack builds such a model from two output modules: an energy head and the
# {class}`~schnetpack.model.Forces` module, which differentiates the predicted
# energy with respect to the positions.
#
# ## Data
#
# The data are 500 frames of an acetic acid trajectory at 500 K, with the EMT
# energy and forces of every frame. The run starts from the EMT minimum.

# %%
acetic_acid = molecule("CH3COOH")
acetic_acid.calc = EMT()
BFGS(acetic_acid, logfile=None).run(fmax=0.01)

atoms = acetic_acid.copy()
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

n_forces = sum(p["forces"].size for p in property_list)
print(f"{len(property_list)} energies and {n_forces} force components")

# %% [markdown]
# The frames go into an ASE database, and a dataset and data module read them as in
# the quickstart (the [data tutorial](data.py) explains each piece). The dataset
# computes the neighbor list once (`preprocess`) and, on every read, removes the
# mean energy and casts to single precision (`transforms`). The data module splits
# the frames into 350 for training, 50 for validation and 100 for testing.

# %%
storage = ASEDBStorage.create(
    os.path.join(workdir, "acetic_acid_emt.db"),
    distance_unit="Ang",
    property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
)
storage.add_systems(property_list, atoms_list)

cutoff = 5.0
dataset = AtomsDataset(
    storage,
    preprocess=[trn.MatScipyNeighborList(cutoff=cutoff)],
    transforms=[trn.RemoveOffsets("energy", remove_mean=True), trn.CastTo32()],
)
datamodule = AtomsDataModule(
    dataset,
    batch_size=32,
    num_train=350,
    num_val=50,
    split_file=os.path.join(workdir, "split.npz"),
)

# %% [markdown]
# ## Model
#
# The model is a PaiNN representation with two output modules. `Atomwise` predicts
# an energy contribution per atom and sums them. `Forces` then computes
# $-\partial E / \partial \mathbf{R}_i$ with PyTorch's autograd. For that, the
# model marks the positions as requiring a gradient before it runs.
#
# In training, `Forces` keeps the graph of this derivative, so that the force loss
# can be differentiated again with respect to the weights. This second derivative
# makes a training step more expensive than one on energies alone.

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
# ## The loss
#
# The loss is a weighted sum of a mean squared error on the energies and one on the
# forces. For a batch of $B$ structures with $N_B$ atoms in total:
#
# $$
# \mathcal{L} = w_E \, \frac{1}{B} \sum_{m=1}^{B} \left( \hat{E}_m - E_m \right)^2
# + w_F \, \frac{1}{3 N_B} \sum_{i=1}^{N_B}
# \left\lVert \hat{\mathbf{F}}_i - \mathbf{F}_i \right\rVert^2,
# $$
#
# where hats mark predictions. Each term comes from one
# {class}`~schnetpack.objectives.ModelOutput`: its `loss_fn` gives the mean squared
# error and its `loss_weight` gives $w_E$ or $w_F$.
#
# The weights set the balance between the two targets. The common choice, also the
# default of SchNetPack's MD17 configs, is $w_E = 0.01$ and $w_F = 0.99$. Most
# weight goes to the forces, which drive an MD run. The energy term is still needed:
# forces fix the energy only up to a constant.

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

# %% [markdown]
# ## Training and keeping the best model
#
# The model after the last epoch is not always the best one. The
# {class}`~schnetpack.lightning.ModelCheckpoint` callback watches the validation
# loss and, whenever it reaches a new minimum, writes two files:
#
# - a Lightning checkpoint in `dirpath`, which holds the whole training state and
#   can resume training;
# - the model alone at `model_path`, ready for predictions. It adds the mean energy
#   back, so it returns energies in the units of the data.

# %%
model_path = os.path.join(workdir, "best_model.pt")
checkpoint = ModelCheckpoint(
    model_path=model_path,
    dirpath=os.path.join(workdir, "checkpoints"),
    monitor="val_loss",
    save_top_k=1,
)

logger = pl.loggers.CSVLogger(workdir)
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    callbacks=[checkpoint],
    # the forces are gradients, so validation and testing need autograd
    inference_mode=False,
    logger=logger,
    default_root_dir=workdir,
    enable_progress_bar=False,
    enable_model_summary=False,
)
trainer.fit(task, datamodule=datamodule)

# %% [markdown]
# The validation errors of every epoch, and the epoch the checkpoint kept:

# %%
with open(os.path.join(logger.log_dir, "metrics.csv")) as f:
    val = [row for row in csv.DictReader(f) if row["val_loss"]]
epochs = [int(row["epoch"]) for row in val]
best = min(val, key=lambda row: float(row["val_loss"]))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(epochs, [float(row["val_energy_MAE"]) for row in val], label="energy (eV)")
ax.plot(epochs, [float(row["val_forces_MAE"]) for row in val], label="forces (eV/Å)")
ax.axvline(int(best["epoch"]), color="gray", linestyle=":", label="kept")
ax.set_xlabel("epoch")
ax.set_ylabel("validation MAE")
ax.set_yscale("log")
ax.legend()
plt.show()

# %% [markdown]
# ## Test errors
#
# `trainer.test` with `ckpt_path="best"` loads the kept checkpoint and evaluates it
# on the test set, which the training never saw:

# %%
trainer.test(task, datamodule=datamodule, ckpt_path="best", verbose=False)
print(f"energy MAE: {trainer.callback_metrics['test_energy_MAE']:.4f} eV")
print(f"force MAE:  {trainer.callback_metrics['test_forces_MAE']:.4f} eV/Å")

# %% [markdown]
# For a closer look, we load the saved model with {func}`~schnetpack.utils.load_model`
# and predict the forces of the test set batch by batch. The model writes its
# predictions into the batch, so the reference forces are taken out first.

# %%
best_model = spk.utils.load_model(model_path)

f_ref, f_pred = [], []
for batch in datamodule.test_dataloader():
    f_ref.append(batch["forces"].numpy())
    f_pred.append(best_model(batch)["forces"].detach().numpy())
f_ref = np.concatenate(f_ref).ravel()
f_pred = np.concatenate(f_pred).ravel()

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.scatter(f_ref, f_pred, s=2)
lim = np.abs(f_ref).max() * 1.05
ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8)
ax.set_xlabel("EMT force component (eV/Å)")
ax.set_ylabel("model force component (eV/Å)")
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Molecular dynamics in ASE
#
# {class}`~schnetpack.interfaces.ase_interface.SpkCalculator` loads the saved model
# as an ASE calculator. A run without a thermostat (NVE) checks the energy
# conservation: we start from the EMT minimum with velocities drawn at 500 K and
# take 500 steps of 0.5 fs with the velocity Verlet integrator. Every step records
# the potential and kinetic energy, and every 100 steps a copy of the structure is
# kept.

# %%
calculator = spk.interfaces.SpkCalculator(
    model=model_path,
    neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff),
    energy_unit="eV",
    position_unit="Ang",
)

atoms = acetic_acid.copy()
atoms.calc = calculator
MaxwellBoltzmannDistribution(atoms, temperature_K=500, rng=rng)
md = VelocityVerlet(atoms, timestep=0.5 * units.fs)

e_pot, e_kin, frames = [], [], []
md.attach(lambda: e_pot.append(atoms.get_potential_energy()), interval=1)
md.attach(lambda: e_kin.append(atoms.get_kinetic_energy()), interval=1)
md.attach(lambda: frames.append(atoms.copy()), interval=100)
md.run(500)
e_pot, e_kin = np.array(e_pot), np.array(e_kin)

# %% [markdown]
# Potential and kinetic energy trade off against each other, and their sum stays
# nearly constant. The plot shows each relative to its value at the start.

# %%
e_tot = e_pot + e_kin
time = 0.5 * np.arange(len(e_pot))  # fs

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(time, e_pot - e_pot[0], label="potential")
ax.plot(time, e_kin - e_kin[0], label="kinetic")
ax.plot(time, e_tot - e_tot[0], label="total")
ax.set_xlabel("time (fs)")
ax.set_ylabel("energy change (eV)")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

print(f"potential energy varies by {np.ptp(e_pot):.3f} eV")
print(f"total energy varies by     {np.ptp(e_tot):.4f} eV")

# %% [markdown]
# Energy conservation shows that the forces are consistent with the model's
# energy, not that they are right. For that, we compare the model with EMT on the
# structures the run kept:

# %%
print(" time (fs)   energy error (eV)   force MAE (eV/Å)")
for step, frame in zip(range(0, 501, 100), frames):
    frame.calc = calculator
    e_model, f_model = frame.get_potential_energy(), frame.get_forces()
    frame.calc = EMT()
    e_emt, f_emt = frame.get_potential_energy(), frame.get_forces()
    print(
        f"{0.5 * step:10.0f}   {e_model - e_emt:17.4f}   "
        f"{np.abs(f_model - f_emt).mean():16.4f}"
    )

# %% [markdown]
# The force errors along the run are of the size of the test error, so the run
# stays in the region the training data covers. Errors that grow along a run show
# that it drifts into structures the training data lacks.
#
# ## The real benchmark: rMD17
#
# The same setup trains on the revised MD17 dataset
# ({class}`~schnetpack.datasets.rMD17`), which holds energies and forces of ten
# small molecules from DFT. On first use it downloads the data, which is why this
# page does not run the cell below. Its native units are kcal/mol and Å; the dataset
# converts the properties to eV when it reads them. The data module draws 1000
# training and 1000 validation structures at random; the rest is the test set.

# %% tags=["skip-execution"]
from schnetpack.datasets import rMD17

dataset = rMD17(
    "rmd17_ethanol.db",
    molecule="ethanol",
    distance_unit="Ang",
    property_units={"energy": "eV", "forces": "eV/Ang"},
    preprocess=[trn.MatScipyNeighborList(cutoff=cutoff)],
    transforms=[trn.RemoveOffsets("energy", remove_mean=True), trn.CastTo32()],
)
datamodule = AtomsDataModule(
    dataset,
    batch_size=10,
    num_train=1000,
    num_val=1000,
    split_file="split.npz",
)

# %% [markdown]
# Model, task and trainer stay as above, with more epochs and a learning rate
# schedule. To compare with published results, use the benchmark's own splits:
# `splitting=rMD17Split(split_id=0)` (from `schnetpack.datasets`, ids 0 to 4) draws
# the training and validation structures from its 1000 training structures, e.g.
# 950 and 50, and tests on its test structures.
#
# The [command line](../getting_started/cli.md) trains the same kind of model from a
# config, the experiment `md17`. [Training faster](../guides/training_speed.md)
# covers the switches that speed up training on a GPU, and the
# [next tutorial](molecular_dynamics.py) runs MD with SchNetPack's own engine
# instead of ASE.

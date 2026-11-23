# %% [markdown]
# # Training a model on molecular properties
#
# This page trains a model that predicts one property, the energy, for many
# different molecules, as one does with the QM9 dataset. The
# [quickstart](../getting_started/quickstart.py) learned the forces of a single
# molecule; new here are per-element reference energies, the SchNet representation,
# and keeping the best model during training.
#
# QM9 holds 134k molecules, too many to train on while the documentation is built.
# So this page uses a small stand-in: 405 structures of small molecules, labelled
# with ASE's EMT calculator, a cheap empirical potential. Training takes about a
# minute on a CPU. The last section shows the same setup on the real QM9.

# %%
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.collections import g2
from ase.optimize import BFGS

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.lightning import AtomisticTask, AtomsDataModule

# everything this page writes goes here; use a directory of your own instead
workdir = tempfile.mkdtemp()
pl.seed_everything(0)

# %% [markdown]
# ## Data
#
# ASE's G2 collection holds the structures of 162 small molecules. We take the 81
# that contain only H, C, N and O, the elements EMT knows, and relax each with EMT.
# Each relaxed molecule and four copies with randomly displaced atoms (standard
# deviation 0.05 Å) make 405 structures.
#
# The target is the energy of each structure relative to its free atoms: the EMT
# energy minus the EMT energies of the isolated atoms. It is about −3 eV per atom,
# so like the energies in QM9 it grows with the size of the molecule.
#
# The structures and energies go into an ASE database, as in the quickstart; the
# [data tutorial](data.py) explains this step in detail.

# %%
free_atom_energy = {}
for z in (1, 6, 7, 8):
    atom = Atoms(numbers=[z])
    atom.calc = EMT()
    free_atom_energy[z] = atom.get_potential_energy()


def relative_energy(atoms):
    """The EMT energy of `atoms` minus that of its free atoms, in eV."""
    free = sum(free_atom_energy[z] for z in atoms.numbers)
    return atoms.get_potential_energy() - free


rng = np.random.default_rng(0)
atoms_list, energies = [], []
for mol in g2:
    if len(mol) == 1 or not set(mol.numbers) <= {1, 6, 7, 8}:
        continue
    mol.calc = EMT()
    BFGS(mol, logfile=None).run(fmax=0.01)
    for k in range(5):
        atoms = mol.copy()
        if k > 0:  # the relaxed structure, then four displaced copies
            atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.calc = EMT()
        energies.append(relative_energy(atoms))
        atoms_list.append(atoms.copy())  # the copy leaves the calculator behind

datapath = os.path.join(workdir, "g2_emt.db")
db = ASEAtomsData.create(
    datapath, distance_unit="Ang", property_unit_dict={"energy": "eV"}
)
db.add_systems([{"energy": np.array([e])} for e in energies], atoms_list)
print(f"{len(atoms_list)} structures")
print(f"energies from {min(energies):.1f} to {max(energies):.1f} eV")

# %% [markdown]
# ## Atom reference energies
#
# Most of the energy of a molecule depends only on which atoms it holds. An
# *atom reference* is one energy per element; summed over the atoms of a structure,
# it is a first guess of the structure's energy. The transform
# {class}`~schnetpack.transform.RemoveOffsets` with `remove_atomrefs=True` subtracts
# this sum from the target, so the network learns only the rest, which is much
# smaller. With `estimate_atomref=True` the references are fitted to the training
# set by least squares. QM9 instead comes with the energies of the free atoms as
# its references (see the last section).
#
# The dataset is built as in the quickstart: its `transforms` hold the neighbor
# list, and the data module's `batch_transforms` remove the offsets and cast every
# batch. The data module splits the 405 structures into 324 for training, 40 for
# validation and the rest for testing. When it is set up, it fits the references on
# the training set. `cache=True` reads this small dataset once, keeps it in memory
# and builds whole batches from it, which is faster than reading it structure by
# structure (see [Training faster](../guides/training_speed.md)).

# %%
cutoff = 5.0

remove_offsets = trn.RemoveOffsets(
    "energy", remove_atomrefs=True, estimate_atomref=True
)
dataset = ASEAtomsData(datapath, transforms=[trn.MatScipyNeighborList(cutoff=cutoff)])
datamodule = AtomsDataModule(
    dataset,
    batch_transforms=[remove_offsets, trn.CastTo32()],
    batch_size=32,
    num_train=324,
    num_val=40,
    split_file=os.path.join(workdir, "split.npz"),
    cache=True,
)
datamodule.setup()

# %% [markdown]
# The fitted references, and how much of the energies they leave for the network:

# %%
atomref = remove_offsets.atomref.numpy()
for z, symbol in [(1, "H"), (6, "C"), (7, "N"), (8, "O")]:
    print(f"{symbol}: {atomref[z]:.2f} eV")

rest = [e - atomref[a.numbers].sum() for e, a in zip(energies, atoms_list)]
print(f"standard deviation of the energies:   {np.std(energies):.2f} eV")
print(f"after subtracting the atom references: {np.std(rest):.2f} eV")

# %% [markdown]
# ## Model
#
# The model has the same parts as in the quickstart, with
# {class}`~schnetpack.model.SchNet` in place of PaiNN. SchNet builds its atom
# features from the distances between atoms alone; PaiNN also passes direction
# vectors between them.
#
# {class}`~schnetpack.model.Atomwise` predicts a contribution per atom. With
# `aggregation_mode="sum"` it adds them up per structure, the right choice for an
# extensive property such as the energy. For an intensive property, one that does
# not grow with the size of the structure, use `"avg"` and pass `is_extensive=False`
# to the offset transforms.
#
# There is no `Forces` module this time: the model predicts only the energy. The
# postprocessors return double precision and add the atom references back. They act
# only when the model is used for prediction, not during training.

# %%
n_atom_basis = 32

model = spk.model.NeuralNetworkPotential(
    representation=spk.model.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=3,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    ),
    input_modules=[spk.model.PairwiseDistances()],
    output_modules=[
        spk.model.Atomwise(
            n_in=n_atom_basis, output_key="energy", aggregation_mode="sum"
        ),
    ],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy", add_atomrefs=True, estimate_atomref=True),
    ],
)

# %% [markdown]
# ## Training
#
# {class}`~schnetpack.objectives.ModelOutput` connects the model's `energy` to the target
# of the same name, with a mean squared error as loss and the mean absolute error
# (MAE) as metric. {class}`~schnetpack.lightning.AtomisticTask` holds the model, the
# outputs and the optimizer.
#
# {class}`~schnetpack.lightning.ModelCheckpoint` keeps the best model: after every
# epoch it compares the validation loss with the best so far. It extends
# Lightning's callback of the same name. Besides the Lightning checkpoint in
# `dirpath`, from which a training can resume, it saves the model alone to the
# folder `model_path`, ready for prediction: `hyperparameters.json` holds its
# classes and their arguments, `model.safetensors` its weights.

# %%
outputs = [
    spk.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )
]
task = AtomisticTask(
    model=model,
    outputs=outputs,
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-3},
)

model_path = os.path.join(workdir, "best_model")
checkpoint = spk.lightning.ModelCheckpoint(
    model_path=model_path,
    dirpath=os.path.join(workdir, "checkpoints"),
    monitor="val_loss",
    save_top_k=1,
)
trainer = pl.Trainer(
    max_epochs=60,
    accelerator="cpu",
    callbacks=[checkpoint],
    logger=False,
    default_root_dir=workdir,
    enable_progress_bar=False,
    enable_model_summary=False,
)
trainer.fit(task, datamodule=datamodule)

# %% [markdown]
# `trainer.test` with `ckpt_path="best"` loads the weights of the best checkpoint
# into the task and computes the metrics on the test set, which the training never
# saw:

# %%
trainer.test(task, datamodule=datamodule, ckpt_path="best", verbose=False)
print(f"test MAE: {trainer.callback_metrics['test_energy_MAE']:.3f} eV")

# %% [markdown]
# Keep in mind that the test set holds other displaced copies of molecules that are
# also in the training set. QM9 has one structure per molecule, so there the test
# set measures the error on new molecules.
#
# ## Predictions
#
# For a closer look, we load the best model from `model_path` and predict the
# energies of the test structures.
# {class}`~schnetpack.interfaces.ase_interface.AtomsConverter` turns a list of ASE `Atoms` into a
# batch, with the neighbor list the model needs. The loaded model adds the atom
# references back, so it predicts the energy itself.

# %%
best_model = spk.load_model(model_path)
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff)
)

test_atoms = [atoms_list[i] for i in datamodule.test_idx]
with torch.no_grad():
    predicted = best_model(converter(test_atoms))["energy"].numpy()
reference = np.array(energies)[datamodule.test_idx]
print(f"MAE: {np.abs(predicted - reference).mean():.3f} eV")

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(
    [reference.min(), reference.max()],
    [reference.min(), reference.max()],
    color="gray",
    linewidth=1,
)
ax.scatter(reference, predicted, s=12)
ax.set_xlabel("EMT energy (eV)")
ax.set_ylabel("predicted energy (eV)")
plt.show()

# %% [markdown]
# The MAE is the one `trainer.test` reported. Next, a molecule outside the G2 set:
# benzonitrile, C₇H₅N, which ASE also knows. We relax it with EMT and compare EMT's
# energy with the model's. {class}`~schnetpack.interfaces.ase_interface.SpkCalculator` wraps the
# model as an ASE calculator, as in the quickstart. The model has no forces, so we
# pass `force_key=None`.

# %%
calculator = spk.interfaces.SpkCalculator(
    model=best_model,
    neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff),
    energy_key="energy",
    force_key=None,
    energy_unit="eV",
    position_unit="Ang",
)

benzonitrile = molecule("C7NH5")
benzonitrile.calc = EMT()
BFGS(benzonitrile, logfile=None).run(fmax=0.01)
print(f"EMT:   {relative_energy(benzonitrile):.2f} eV")

benzonitrile.calc = calculator
print(f"model: {benzonitrile.get_potential_energy():.2f} eV")

# %% [markdown]
# ## The real QM9
#
# On QM9 the steps are the same. The {class}`~schnetpack.datasets.QM9` class
# downloads the dataset on first use and writes it into an ASE database. Its
# properties have names on the class, e.g. `QM9.U0` for the energy at 0 K.
#
# QM9 comes with the energies of the free atoms as atom references, stored in the
# database's metadata (`qm9.atomrefs`). So `RemoveOffsets` takes those and needs no
# fit. With `remove_mean=True` it also removes the mean energy per atom that the
# references leave; `AddOffsets` adds both back. The cell below is not run when the
# documentation is built. From the command line, `spk train experiment=qm9_atomwise`
# trains the same kind of model (see [Configs](../guides/configs.md)).

# %% tags=["skip-execution"]
from schnetpack.datasets import QM9

qm9 = QM9(
    os.path.join(workdir, "qm9.db"),  # keep the download in a directory of your own
    load_properties=[QM9.U0],
    property_units={QM9.U0: "eV"},
    transforms=[trn.MatScipyNeighborList(cutoff=5.0)],
)
print(qm9.atomrefs[QM9.U0][[1, 6, 7, 8, 9]])  # free H, C, N, O and F atoms, in eV

datamodule = AtomsDataModule(
    qm9,
    batch_transforms=[
        trn.RemoveOffsets(QM9.U0, remove_atomrefs=True, remove_mean=True),
        trn.CastTo32(),
    ],
    batch_size=100,
    num_train=110000,
    num_val=10000,
    split_file=os.path.join(workdir, "qm9_split.npz"),
    num_workers=4,
)
model = spk.model.NeuralNetworkPotential(
    representation=spk.model.SchNet(
        n_atom_basis=128,
        n_interactions=6,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0),
    ),
    input_modules=[spk.model.PairwiseDistances()],
    output_modules=[spk.model.Atomwise(n_in=128, output_key=QM9.U0)],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True),
    ],
)
task = AtomisticTask(
    model=model,
    outputs=[
        spk.ModelOutput(
            name=QM9.U0,
            loss_fn=torch.nn.MSELoss(),
            metrics={"MAE": torchmetrics.MeanAbsoluteError()},
        )
    ],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 5e-4},
)
trainer = pl.Trainer(
    accelerator="gpu",
    callbacks=[
        spk.lightning.ModelCheckpoint(
            model_path=os.path.join(workdir, "qm9_model"), monitor="val_loss"
        )
    ],
    default_root_dir=workdir,
)
trainer.fit(task, datamodule=datamodule)

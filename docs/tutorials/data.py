# %% [markdown]
# # Preparing and loading your data
#
# This page shows how SchNetPack holds atomistic data: what one structure looks
# like, how to open an existing dataset and write your own, how transforms prepare
# each structure, and how a data module splits a dataset and batches it. Nothing is
# trained here; the [force field tutorial](force_fields.py) does that.
#
# The data is small and built in place, so that the page runs in a few seconds:
# 20 QM9 molecules from SchNetPack's test files, and a few molecules labelled with
# ASE's EMT calculator, a cheap empirical potential. The last section shows how to
# load the full QM9 benchmark instead.

# %%
import os
import tempfile

import numpy as np
import pytorch_lightning as pl
import torch
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT

import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.lightning import AtomsDataModule

# everything this page writes goes here; use a directory of your own instead
workdir = tempfile.mkdtemp()

# %% [markdown]
# ## One structure
#
# SchNetPack passes a structure around as a dictionary of tensors. The keys are
# defined in {mod}`schnetpack.properties`: keys with a leading underscore describe
# the structure, plain names such as `energy` and `forces` are its properties. Here
# we build such a dictionary by hand for a water molecule, labelled with EMT.

# %%
water = molecule("H2O")
water.calc = EMT()

structure = {
    properties.Z: torch.tensor(water.numbers),  # atomic numbers
    properties.R: torch.tensor(water.positions),  # positions in Å
    properties.cell: torch.tensor(water.cell.array)[None],  # unit cell in Å
    properties.pbc: torch.tensor(water.pbc),  # periodic along each cell vector?
    properties.n_atoms: torch.tensor([len(water)]),
    "energy": torch.tensor([water.get_potential_energy()]),  # eV
    "forces": torch.tensor(water.get_forces()),  # eV/Å
}
for key, value in structure.items():
    print(f"{key:16s} shape {tuple(value.shape)}")

# %% [markdown]
# Per-atom entries such as the positions and forces have one row per atom.
# Per-structure entries such as the energy, the cell and the atom count have a first
# dimension of one, so that the structures of a batch can be concatenated along it.
# A molecule has a zero cell and no periodic boundary conditions. The
# [glossary](../concepts/glossary.md) lists all keys.
#
# ## Opening a dataset
#
# A dataset file is an ASE database (`.db`). Next to the structures, its metadata
# holds the unit of every property and of the positions, and optionally atom
# references (below). {class}`~schnetpack.data.ASEAtomsData` opens such a file and
# returns one structure per index, as a dictionary like the one above.
#
# The file `tests/testdata/test_qm9.db` of the SchNetPack repository holds 20
# molecules of QM9 with all 15 QM9 properties.

# %%
qm9_path = "../../tests/testdata/test_qm9.db"
dataset = ASEAtomsData(qm9_path)

print("structures:", len(dataset))
print("distance unit:", dataset.distance_unit)
print("properties and units:")
for name, unit in dataset.units.items():
    print(f"  {name:26s} {unit}")

# %% [markdown]
# By default a structure is read with all its properties. `load_properties` reads
# only some of them, and `subset_idx` takes only some of the structures. The key
# `_idx` gives the index of a structure in the file.

# %%
dataset = ASEAtomsData(
    qm9_path,
    load_properties=["energy_U0", "homo"],
    subset_idx=[0, 5, 10, 15],
)
print("structures:", len(dataset))
print("loaded properties:", dataset.load_properties)

example = dataset[1]
print("index in the file:", example[properties.idx].item())
for key, value in example.items():
    print(f"  {key:16s} shape {tuple(value.shape)}")

# %% [markdown]
# The file keeps the units it was written with: QM9 stores energies in Hartree.
# `property_units` and `distance_unit` convert the properties and positions when a
# structure is read. Here we read the energies in eV and the positions in Bohr.

# %%
dataset_ev = ASEAtomsData(
    qm9_path,
    load_properties=["energy_U0", "homo"],
    property_units={"energy_U0": "eV", "homo": "eV"},
    distance_unit="Bohr",
)
print("units:", {name: dataset_ev.units[name] for name in dataset_ev.load_properties})
print("distance unit:", dataset_ev.distance_unit)

stored, converted = ASEAtomsData(qm9_path)[0], dataset_ev[0]
print(
    f"U0:  {stored['energy_U0'].item():.4f} Ha = {converted['energy_U0'].item():.2f} eV"
)
print(
    f"y of the first atom: {stored[properties.R][0, 1].item():.4f} Å = "
    f"{converted[properties.R][0, 1].item():.4f} Bohr"
)

# %% [markdown]
# QM9 also stores atom references: the energy of each isolated atom, computed with
# the same method as the molecules. They are a list indexed by atomic number and are
# converted to the requested units as well. Subtracting them from a molecule's
# energy leaves its atomization energy, which is much smaller and easier to learn.

# %%
refs = dataset_ev.atomrefs["energy_U0"]
for symbol, z in [("H", 1), ("C", 6), ("N", 7), ("O", 8), ("F", 9)]:
    print(f"{symbol}: {refs[z].item():9.2f} eV")

# %% [markdown]
# ## Writing your own dataset
#
# Your own data goes into a new ASE database, written through
# {class}`~schnetpack.data.ASEAtomsData`. Its `add_systems` takes two lists of equal
# length: for each structure a dictionary from property name to numpy array, and the
# structures as ASE `Atoms`.
#
# As a stand-in for quantum chemistry, we take six small molecules, displace their
# atoms at random by about 0.05 Å, ten times each, and label every structure with
# the EMT energy and forces.

# %%
rng = np.random.default_rng(0)
atoms_list, property_list = [], []
for name in ["H2O", "NH3", "CH4", "CH3OH", "C2H6", "CH3CH2OH"]:
    for _ in range(10):
        atoms = molecule(name)
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.calc = EMT()
        property_list.append(
            {
                "energy": np.array([atoms.get_potential_energy()]),  # shape (1,)
                "forces": atoms.get_forces(),  # shape (n_atoms, 3)
            }
        )
        atoms_list.append(atoms.copy())  # without the calculator

print(len(atoms_list), "structures")
print(f"energy of the first structure: {property_list[0]['energy'][0]:.3f} eV")

# %% [markdown]
# The atom references must come from the same method as the data, so we compute
# the EMT energy of each isolated atom. They are stored per property, as a list
# indexed by atomic number. If your data has none, leave them out.

# %%
atomrefs = np.zeros(100)
for symbol in ["H", "C", "N", "O"]:
    atom = Atoms(symbol)
    atom.calc = EMT()
    atomrefs[atom.numbers[0]] = atom.get_potential_energy()

# %% [markdown]
# `ASEAtomsData.create` writes a new, empty database with the units and atom
# references in its metadata and returns a dataset on it; `add_systems` adds the
# structures. Units are strings such as `"eV"`, `"kcal/mol"`, `"Ang"` or `"Bohr"`.
# `create` refuses to overwrite an existing file; to add structures later, open it
# with `ASEAtomsData(path)` and call `add_systems` again.

# %%
db_path = os.path.join(workdir, "molecules_emt.db")
db = ASEAtomsData.create(
    db_path,
    distance_unit="Ang",
    property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
    atomrefs={"energy": atomrefs.tolist()},
)
db.add_systems(property_list, atoms_list)

print(len(db), "structures, units:", db.units)
print("atom reference of C:", db.atomrefs["energy"][6].item(), "eV")

# %% [markdown]
# ## Transforms
#
# Transforms are PyTorch modules that take and return the dictionary. They run in
# two places:
#
# - the dataset's `transforms` act on one structure, every time it is read, such as
#   the neighbor list;
# - the loader's `batch_transforms` act on a batch of structures, such as removing
#   energy offsets, casting to single precision or adding noise (next section).
#
# The neighbor list holds all pairs of atoms closer than a cutoff, here 5 Å.

# %%
cutoff = 5.0  # Å
dataset = ASEAtomsData(db_path, transforms=[trn.MatScipyNeighborList(cutoff=cutoff)])

stored = ASEAtomsData(db_path)[0]  # as stored, without transforms
example = dataset[0]
print("added keys:", sorted(set(example) - set(stored)))
print("idx_i:", example[properties.idx_i].tolist())
print("idx_j:", example[properties.idx_j].tolist())

# %% [markdown]
# The neighbor list adds `_idx_i` and `_idx_j`, the center and neighbor atom of
# every pair, and `_offsets`, the cell offset of every pair (zero for a molecule).
# Water has three atoms, so all six ordered pairs are within the cutoff. A
# periodic crystal works the same way; its pairs then include periodic images.
#
# ## Splits and batches
#
# {class}`~schnetpack.lightning.AtomsDataModule` prepares a dataset for training.
# It splits the dataset at random into training, validation and test sets,
# computes the statistics of the training set that the transforms need, and builds
# a data loader for each set. It is a PyTorch Lightning `LightningDataModule`, so a
# Lightning `Trainer` calls its `setup()`; here we call it by hand.
#
# Its `batch_transforms` run on every batch: `RemoveOffsets` subtracts the atom
# references and the mean energy per atom, and `CastTo32` casts all floating point
# tensors to single precision.

# %%
pl.seed_everything(0)
datamodule = AtomsDataModule(
    dataset,
    batch_transforms=[
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=True),
        trn.CastTo32(),
    ],
    batch_size=4,
    num_train=40,
    num_val=10,  # the remaining 10 structures are the test set
    split_file=os.path.join(workdir, "split.npz"),
)
datamodule.setup()

print("training:  ", len(datamodule.train_dataset))
print("validation:", len(datamodule.val_dataset))
print("test:      ", len(datamodule.test_dataset))

# %% [markdown]
# `num_train` and `num_val` also take fractions of the dataset, e.g. `0.8`. The
# split is written to `split_file`. Another run with the same file reads the split
# from it instead of drawing a new one, so the test set stays the same when you
# train again. The statistics are saved next to it.
#
# For an extensive property such as the energy, the statistics are taken per atom,
# after the atom references are removed. `RemoveOffsets` reads them from the data
# module and subtracts the mean per atom times the number of atoms. What is left is
# a small number for the network to learn. A dataset without atom references can
# use `RemoveOffsets(..., estimate_atomref=True)`, which fits them to the training
# set by least squares.

# %%
mean, std = datamodule.get_stats("energy", divide_by_atoms=True, remove_atomref=True)
print(f"energy per atom after atom references: mean {mean:.3f} eV, std {std:.3f} eV")

# %% [markdown]
# The data loaders read the structures of each set, join them into batches and pass
# each batch through the batch transforms. The training loader shuffles the
# structures in every epoch.

# %%
batch = next(iter(datamodule.train_dataloader()))
for key, value in batch.items():
    print(f"{key:16s} shape {tuple(value.shape)}")
print("energies after RemoveOffsets:", batch["energy"])

# %% [markdown]
# A batch joins its structures along the atom dimension: the positions of all four
# structures are one tensor with one row per atom, without padding. `_n_atoms`
# gives the number of atoms of each structure, and `_idx_m` the structure of each
# atom. The neighbor indices `_idx_i` and `_idx_j` now count the atoms of the whole
# batch; `_idx_i_local` and `_idx_j_local` keep the numbering within each
# structure.

# %%
print("atoms per structure:", batch[properties.n_atoms].tolist())
print("structure of each atom:", batch[properties.idx_m].tolist())

# %% [markdown]
# `datamodule.val_dataloader()` and `datamodule.test_dataloader()` give the
# batches of the other two sets, in a fixed order.
#
# ## Reading every structure once
#
# The dataset's transforms run again every time a structure is read. With
# `cache=True` the data module reads every structure once instead, with its
# transforms, into an {class}`~schnetpack.data.AtomsCache` in memory, and the
# loaders slice whole batches from it: no worker processes, no reads per structure.
# This is the fast route for training on a GPU; see
# [Training faster](../guides/training_speed.md). The dataset's transforms then run
# only once, so they must give the same result every time: noise and offsets belong
# among the batch transforms.

# %%
cached = AtomsDataModule(
    dataset,
    batch_transforms=[
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=True),
        trn.CastTo32(),
    ],
    batch_size=4,
    num_train=40,
    num_val=10,
    split_file=os.path.join(workdir, "split.npz"),  # the same split as above
    cache=True,
)
cached.setup()
batch = next(iter(cached.val_dataloader()))
print("atoms per structure:", batch[properties.n_atoms].tolist())
print("real atoms:", batch[properties.atom_mask].tolist())

# %% [markdown]
# Batches from a cache are padded to fixed sizes, so that every training step has
# the same shapes. The padding atoms form one extra structure at the end, with
# `_atom_mask` and `_structure_mask` 0; the training task leaves them out of the
# loss and the metrics.
#
# ## The QM9 benchmark
#
# SchNetPack downloads several benchmark datasets on its own, such as
# {class}`~schnetpack.datasets.QM9` with its 133,885 molecules. These classes are
# subclasses of `ASEAtomsData` and take its arguments, such as `load_properties`,
# `property_units` and `transforms`. If the database at the given path does not
# exist, they download the data and write it there first.
#
# The cell below shows how you would load QM9 and prepare it for training. It is
# not run when this page is built, since the download is large; the page uses the
# small files above instead.

# %% tags=["skip-execution"]
from schnetpack.datasets import QM9

qm9 = QM9(
    "qm9.db",  # downloaded and converted on first use
    remove_uncharacterized=True,  # leave out the 3,054 molecules that failed a check
    load_properties=[QM9.U0],
    property_units={QM9.U0: "eV"},
    transforms=[trn.MatScipyNeighborList(cutoff=5.0)],
)
qm9_datamodule = AtomsDataModule(
    qm9,
    batch_transforms=[
        trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
        trn.CastTo32(),
    ],
    batch_size=100,
    num_train=110000,
    num_val=10000,
    split_file="split_qm9.npz",
    num_workers=4,  # read and transform the structures in 4 processes
)
qm9_datamodule.setup()

# %% [markdown]
# # Preparing and loading your data
# This tutorial introduces how SchNetPack stores and loads data.
# Before we can start training neural networks with SchNetPack, we need to prepare our data.
# This is because SchNetPack has to stream the reference data from disk during training in order to be able to handle large datasets.
# Therefore, it is crucial to use data format that allows for fast random read access.
# We found that the [ASE database format](https://wiki.fysik.dtu.dk/ase/ase/db/db.html) fulfills this criterion perfectly.
# To further improve the performance, we internally encode properties in binary.
# However, as long as you only access the ASE database via the provided SchNetPack `ASEAtomsData` class, you don't have to worry about that.

# %%
import os
import urllib.request

import numpy as np
from ase import Atoms

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList

# %% [markdown]
# ## Predefined datasets
# SchNetPack supports several benchmark datasets that can be used without preparation.
# Each one can be accessed using a corresponding class that inherits from `AtomsDataModule` (a specialized PyTorchLightning `DataModule`), which supports automatic download, conversion and partitioning. Here, we show how to use these data sets at the example of the QM9 benchmark.
#
# First, we have to import the dataset class and instantiate it. This will automatically download the data to the specified location.

# %%
qm9data = QM9(
    "./qm9.db",
    batch_size=10,
    num_train=110000,
    num_val=10000,
    split_file="./split_qm9.npz",
    transforms=[ASENeighborList(cutoff=5.0)],
)
qm9data.prepare_data()
qm9data.setup()

# %% [markdown]
# Neighbors are collected using neighborlists that can be passed to the `AtomsDataModule` as a preprocessing transform. These are applied to the molecules before they are batched in the data loader. We supply different environment providers using a cutoff (e.g., `AseEnvironmentProvider`, `TorchEnvironmentProvider`) that are able to handle larger molecules and periodic boundary conditions.
#
# Let's have a closer look at this dataset.
# We can find out how large it is and which properties it supports:

# %%
print("Number of reference calculations:", len(qm9data.dataset))
print("Number of train data:", len(qm9data.train_dataset))
print("Number of validation data:", len(qm9data.val_dataset))
print("Number of test data:", len(qm9data.test_dataset))
print("Available properties:")

for p in qm9data.dataset.available_properties:
    print("-", p)

# %% [markdown]
# We can load data points  using zero-base indexing. The result is a dictionary containing the geometry and properties:

# %%
example = qm9data.dataset[0]
print("Properties:")

for k, v in example.items():
    print("-", k, ":", v.shape)

# %% [markdown]
# We see that all available properties have been loaded as torch tensors with the given shapes. Keys with an underscore indicate that these names are reserved for internal use. This includes the geometry (`_n_atoms`, `_atomic_numbers`, `_positions`), the index within the dataset (`_idx`) as well as information about neighboring atoms and periodic boundary conditions (`_cell`, `_pbc`).
#
#
# We can iterate the dataset partitions as follows:

# %%
for batch in qm9data.val_dataloader():
    print(batch.keys())
    break

# %% [markdown]
# We see that additional keys have been added by the neighborlist transform defined above. These are the relative positions (`_Rij`) and neighbor indices (`_idx_i`, `_idx_j`). Since differrent systems can have different numbers of atoms, we don't use separate dimensions for systems and atoms (i.e. shape [n_systems, n_atoms, ...]), but store the atoms of all systems in a single dimension (i.e. shape [n_all_atoms, ...]). Therefore, we additionally need to store the indices of the corresponding system for each atom in a batch (`idx_m`). This avoids the padding and masking that was required in previous versions of SchNetPack. The indices look as follows:

# %%
print("System index:", batch["_idx_m"])
print("Center atom index:", batch["_idx_i"])
print("Neighbor atom index:", batch["_idx_j"])

# %% [markdown]
# All property names are pre-defined as class-variable for convenient access:

# %%
print("Total energy at 0K:", batch[QM9.U0])
print("HOMO:", batch[QM9.homo])

# %% [markdown]
# ## Preparing your own data
# In the following we will create an ASE database from our own data.
# For this tutorial, we will use a dataset containing a molecular dynamics (MD) trajectory of ethanol, which can be downloaded [here](http://quantum-machine.org/gdml/data/xyz/ethanol_dft.zip).

# %%
if not os.path.exists("./md17_uracil.npz"):
    urllib.request.urlretrieve(
        "http://quantum-machine.org/gdml/data/npz/md17_uracil.npz",
        "./md17_uracil.npz",
    )

# %% [markdown]
# The data set is in Numpy format.
# In the following, we show how this data can be parsed and converted for use in SchNetPack, so that you apply this to any other data format.
#
# First, we need to parse our data. For this we use the IO functionality supplied by ASE.
# In order to create a SchNetPack DB, we require a **list of ASE `Atoms` objects** as well as a corresponding **list of dictionaries** `[{property_name1: property1_molecule1}, {property_name1: property1_molecule2}, ...]` containing the mapping from property names to values.

# %%
# load atoms from npz file. Here, we only parse the first 10 molecules
data = np.load("./md17_uracil.npz")

numbers = data["z"]
atoms_list = []
property_list = []
for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
    ats = Atoms(positions=positions, numbers=numbers)
    properties = {"energy": energies, "forces": forces}
    property_list.append(properties)
    atoms_list.append(ats)

print("Properties:", property_list[0])

# %% [markdown]
# Once we have our data in this format, it is straightforward to create a new SchNetPack DB and store it.

# %%
# %rm './new_dataset.db'
new_dataset = ASEAtomsData.create(
    "./new_dataset.db",
    distance_unit="Ang",
    property_unit_dict={"energy": "kcal/mol", "forces": "kcal/mol/Ang"},
)
new_dataset.add_systems(property_list, atoms_list)

# %% [markdown]
# To get a better initialization of the network and avoid numerical issues, we often want to make use of simple statistics of our target properties. The most simple approach is to subtract the mean value of our target property from the labels before training such that the neural networks only have to learn the difference from the mean prediction. A more sophisticated approach is to use so-called atomic reference values that provide basic statistics of our target property based on the atom types in a structure. This is especially useful for extensive properties such as the energy, where the single atom energies contribute a major part to the overall value. If your data comes with atomic reference values, you can add them to the metadata of your `ase` database. The statistics have to be stored in a dictionary with the property names as keys and the atomic reference values as lists where the list indices match the atomic numbers. For further explanation please have a look at the [QM9 tutorial](https://schnetpack.readthedocs.io/en/latest/tutorials/tutorial_02_qm9.html).
#
# Here is an example:

# %%
# calculate this at the same level of theory as your data
atomref = {"energy": [314.0, 0.0, 0.0, 0.0]}  # atomref value for hydrogen: 314.0

# the supplied list is ordered by atomic number, e.g.:
atomref_hydrogen = atomref["energy"][1]

# dataset = ASEAtomsData.create(
#     './new_dataset.db',
#     distance_unit='Ang',
#     property_unit_dict={'energy':'kcal/mol'},
#     atomref=atomref
# )

# %% [markdown]
# In our concrete case, we only have an MD trajectory of a single system. Therefore, we don't need to specify an atomref, since removing the average energy will working as well.

# %% [markdown]
# Now we can have a look at the data in the same way we did before for QM9:

# %%
print("Number of reference calculations:", len(new_dataset))
print("Available properties:")

for p in new_dataset.available_properties:
    print("-", p)
print()

example = new_dataset[0]
print("Properties of molecule with id 0:")

for k, v in example.items():
    print("-", k, ":", v.shape)

# %% [markdown]
# The same way, we can store multiple properties, including atomic properties such as forces, or tensorial properties such as polarizability tensors.

# %% [markdown]
# ## Using your data for training
# We have now used the class `ASEAtomsData` to create a new `ase` database for our custom data. `schnetpack.data.ASEAtomsData` is a subclass of `pytorch.data.Dataset` and could be utilized for training models with `pytorch`. However, we use `pytorch-lightning` to conveniently handle the training procedure for us. This requires us to wrap the dataset in a [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). We provide a general purpose `AtomsDataModule` for atomic systems in `schnetpack.data.datamodule.AtomsDataModule`. The data module will handle the unit conversion, splitting, batching and the preprocessing of the data with `transforms`. We can instantiate the data module for our custom dataset with:

# %%
custom_data = spk.data.AtomsDataModule(
    "./new_dataset.db",
    batch_size=10,
    distance_unit="Ang",
    property_units={"energy": "kcal/mol", "forces": "kcal/mol/Ang"},
    num_train=1000,
    num_val=100,
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32(),
    ],
    num_workers=1,
    pin_memory=True,  # set to false, when not using a GPU
)
custom_data.prepare_data()
custom_data.setup()

# %% [markdown]
# Please note that for the general case it makes sense to use your dataset within command line interface (see: [here](https://schnetpack.readthedocs.io/en/latest/userguide/configs.html)). For some benchmark datasets we provide data modules with download functions and more utilities in `schnetpack.data.datasets`. Further examples on how to use the data modules is provided in the following sections.
#

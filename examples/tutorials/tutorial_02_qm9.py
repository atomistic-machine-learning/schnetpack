# ---
# jupyter:
#   nbsphinx:
#     execute: never
# ---

# %% [markdown]
# # Training a neural network on QM9
#
# This tutorial will explain how to use SchNetPack for training a model
# on the QM9 dataset and how the trained model can be used for further applications.
#
# First, we import the necessary modules and create a new directory for the data and our model.

# %%
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from ase import Atoms

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.datasets import QM9
from schnetpack.utils.compatibility import load_model

# %%
qm9tut = "./qm9tut"
if not os.path.exists("qm9tut"):
    os.makedirs(qm9tut)

# %% [markdown]
# ## Loading the data
#
# As explained in the [previous tutorial](tutorial_01_preparing_data.py), datasets in SchNetPack are loaded with the `AtomsLoader` class or one of the sub-classes that are specialized for common benchmark datasets.
# The `QM9` dataset class will download and convert the data. We will only use the inner energy at 0K `U0`, so all other properties do not need to be loaded:

# %%
# %rm split.npz

qm9data = QM9(
    "./qm9.db",
    batch_size=100,
    num_train=1000,
    num_val=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
        trn.CastTo32(),
    ],
    property_units={QM9.U0: "eV"},
    num_workers=1,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True,  # set to false, when not using a GPU
    load_properties=[QM9.U0],  # only load U0 property
)
qm9data.prepare_data()
qm9data.setup()

# %% [markdown]
# The dataset is downloaded and partitioned automatically. PyTorch `DataLoader`s can be obtained using `qm9data.train_dataloader()`, `qm9data.val_dataloader()` and `qm9data.test_dataloader()`.
#
# Before building the model, we remove offsets from the energy for good initial conditions. We will get this from the training dataset. Above, this is done automatically by the `RemoveOffsets` transform.
# In the following we show what happens under the hood.
# For QM9, we also have single-atom reference values stored in the metadata:

# %%
atomrefs = qm9data.train_dataset.atomrefs
print("U0 of hyrogen:", atomrefs[QM9.U0][1].item(), "eV")
print("U0 of carbon:", atomrefs[QM9.U0][6].item(), "eV")
print("U0 of oxygen:", atomrefs[QM9.U0][8].item(), "eV")

# %% [markdown]
# These can be used together with the mean and standard deviation of the energy per atom to initialize the model with a good guess of the energy of a molecule. When calculating these statistics, we pass the atomref to take into account, that the model will add these atomrefs to the predicted energy later, so that this part of the energy does not have to be considered in the statistics, i.e.
# \begin{equation}
# \mu_{U_0} = \frac{1}{n_\text{train}} \sum_{n=1}^{n_\text{train}} \left( U_{0,n} - \sum_{i=1}^{n_{\text{atoms},n}} U_{0,Z_{n,i}} \right)
# \end{equation}
# for the mean and analogously for the standard deviation. In this case, this corresponds to the mean and std. dev of the *atomization energy* per atom.

# %%
means, stddevs = qm9data.get_stats(QM9.U0, divide_by_atoms=True, remove_atomref=True)
print("Mean atomization energy / atom:", means.item())
print("Std. dev. atomization energy / atom:", stddevs.item())

# %% [markdown]
# ## Setting up the model
#
# Next, we need to build the model and define how it should be trained.
#
# In SchNetPack, a neural network potential usually consists of three parts:
#
# 1. A list of input modules that prepare the batched data before the building the representation.
#    This includes, e.g., the calculation of pairwise distances between atoms based on neighbor indices or add auxiliary
#    inputs for response properties.
# 2. The representation which either constructs atom-wise features, e.g. with SchNet or PaiNN.
# 3. One or more output modules for property prediction.
#
# Here, we use the `SchNet` representation with 3 interaction layers, a 5 Angstrom cosine cutoff with pairwise distances
# expanded on 20 Gaussians and 30 atomwise features and convolution filters, since we only have a few
# training examples. Then, we use an `Atomwise` module to predict the inner energy $U_0$ by summing over atom-wise
# energy contributions.

# %%
cutoff = 5.0
n_atom_basis = 30

pairwise_distance = (
    spk.atomistic.PairwiseDistances()
)  # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis,
    n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)
pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_U0],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True),
    ],
)

# %% [markdown]
# The last argument here is a list of postprocessors that will only be used if `nnpot.inference_mode=True` is set.
# It will not be used in training or validation, but only for predictions.
# Here, this is used to deal with numerical accuracy and normalization of model outputs:
# To make training easier, we have subtracted single atom energies as well as the mean energy per atom
# in the preprocessing (see above).
# This does not matter for the loss, but for the final prediction we want to get the real energies.
# Additionally, we have removed the energy offsets *before* casting to float32 in the preprocessor.
# This avoids loss of numerical precision.
# Analog to this, we also have to first cast to float64, before re-adding the offsets in the post-processor
#
# The output modules store the prediction in a dictionary under the `output_key` (here: `QM9.U0`), which is connected to
# a target property with loss functions and evaluation metrics using the `ModelOutput` class:

# %%
output_U0 = spk.task.ModelOutput(
    name=QM9.U0,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)

# %% [markdown]
# By default, the target is assumed to have the same name as the output. Otherwise, a different `target_name`
# has to be provided.
# Here, we already gave the output the same name as the target in the dataset (`QM9.U0`).
# In case of multiple outputs, the full loss is a weighted sum of all output losses.
# Therefore, it is possible to provide a `loss_weight`, which we here just set to 1.
#
# All components defined above are then passed to `AtomisticTask`, which is a sublass of
# [`LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).
# This connects the model and training process and can then be passed to the PyTorch Lightning `Trainer`.

# %%
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_U0],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
)

# %% [markdown]
# ## Training the model
#
# Now, the model is ready for training. Since we already defined all necessary components, the only thing left to do is
# passing it to the PyTorch Lightning `Trainer` together with the data module.
#
# Additionally, we can provide callbacks that take care of logging, checkpointing etc.

# %%
logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(qm9tut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss",
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=qm9tut,
    max_epochs=3,  # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=qm9data)

# %% [markdown]
# The `ModelCheckpoint` of SchNetPack is equivalent to that in PyTorch Lightning,
# except that we also store the best inference model. We will show how to use this in the next section.
#
# You can have a look at the training log using Tensorboard:
# ```
# tensorboard --logdir=qm9tut/lightning_logs
# ```
#
#

# %% [markdown]
# ## Inference
#
# Having trained a model for QM9, we are going to use it to obtain some predictions.
# First, we need to load the model. The `Trainer` stores the best model in the model directory which can be loaded using PyTorch:

# %%
best_model = load_model(os.path.join(qm9tut, "best_inference_model"), device="cpu")

# %% [markdown]
# We can use the test dataloader from the QM( data to obtain a batch of molecules and apply the model:

# %%
for batch in qm9data.test_dataloader():
    result = best_model(batch)
    print("Result dictionary:", result)
    break

# %% [markdown]
# If your data is not already in SchNetPack format, a convenient way is to use ASE atoms with the
# provided `AtomsConverter`:

# %%
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32
)

# %%
numbers = np.array([6, 1, 1, 1, 1])
positions = np.array(
    [
        [-0.0126981359, 1.0858041578, 0.0080009958],
        [0.002150416, -0.0060313176, 0.0019761204],
        [1.0117308433, 1.4637511618, 0.0002765748],
        [-0.540815069, 1.4475266138, -0.8766437152],
        [-0.5238136345, 1.4379326443, 0.9063972942],
    ]
)
atoms = Atoms(numbers=numbers, positions=positions)

# %%
inputs = converter(atoms)

print("Keys:", list(inputs.keys()))

pred = best_model(inputs)

print("Prediction:", pred[QM9.U0])

# %% [markdown]
# Alternatively, one can use the `SpkCalculator` as an interface to ASE. The calculator requires the path to a trained model and a neighborlist as input. In addition, the names and units of properties used in the model (e.g. the energy) should be provided. Precision and device can be set via the `dtype` and `device` keywords:

# %%
calculator = spk.interfaces.SpkCalculator(
    model=os.path.join(qm9tut, "best_inference_model"),  # path to model
    neighbor_list=trn.ASENeighborList(cutoff=5.0),  # neighbor list
    energy_key=QM9.U0,  # name of energy property in model
    force_key=None,
    energy_unit="eV",  # units of energy property
    device="cpu",  # device for computation
)
atoms.calc = calculator
print("Prediction:", atoms.get_total_energy())

# %% [markdown]
# The calculator automatically converts the prediction of the given unit to internal ASE units, which is `eV`
# for the energy.
# Using the calculator interface makes more sense if you have trained SchNet for a potential energy surface.
# In the next tutorials, we will show how to learn potential energy surfaces and forces field as well as performing
# molecular dynamics simulations with SchNetPack.

# ---
# jupyter:
#   nbsphinx:
#     execute: never
# ---

# %% [markdown]
# # Training a model on forces and energies
#
# In addition to the energy, machine learning models can also be used to model molecular forces.
# These are $N_\mathrm{atoms} \times 3$ arrays describing the Cartesian force acting on each atom due to the overall
# (potential) energy. They are formally defined as the negative gradient of the energy $E_\mathrm{pot}$ with respect to
# the nuclear positions $\mathbf{R}$
#
# \begin{equation}
# \mathbf{F}^{(\alpha)} = -\frac{\partial E_\mathrm{pot}}{\partial \mathbf{R}^{(\alpha)}},
# \end{equation}
#
# where $\alpha$ is the index of the nucleus.
#
# The above expression offers a straightforward way to include forces in machine learning models by simply defining a
# model for the energy and taking the appropriate derivatives.
# The resulting model can directly be trained on energies and forces.
# Moreover, in this manner energy conservation and the correct behaviour under rotations of the molecule is guaranteed.
#
# Using forces in addition to energies to construct a machine learning model offers several advantages.
# Accurate force predictions are important for molecular dynamics simulations, which will be covered in the subsequent
# tutorial. Forces also encode a greater wealth of information than the energies.
# For every molecule, only one energy is present, while there are $3N_\mathrm{atoms}$ force entries.
# This property, combined with the fact that reference forces can be computed at the same cost as energies, makes models
# trained on forces and energies very data efficient.
#
# In the following, we will show how to train such force models and how to use them in practical applications.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from ase import Atoms, io

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.datasets import MD17
from schnetpack.utils.compatibility import load_model

# %% [markdown]
# ## Preparing the data
#
# The process of preparing the data is similar to the tutorial on [QM9](tutorial_02_qm9.py). We begin by importing all
# relevant packages and generating a directory for the tutorial experiments.

# %%
forcetut = "./forcetut"
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

# %% [markdown]
# Next, the data needs to be loaded from a suitable dataset.
# For convenience, we use the MD17 dataset class provided in SchNetPack, which automatically downloads and builds suitable
# databases containing energies and forces for a range of small organic molecules.
# In this case, we use the ethanol molecule as an example.

# %%
ethanol_data = MD17(
    os.path.join(forcetut, "ethanol.db"),
    molecule="ethanol",
    batch_size=10,
    num_train=1000,
    num_val=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32(),
    ],
    num_workers=1,
    pin_memory=True,  # set to false, when not using a GPU
)
ethanol_data.prepare_data()
ethanol_data.setup()

# %% [markdown]
# The data is split into training (1000 points), validation (1000 points) and test set (remainder).
# Once again, we subtract the mean of the energies in the training data with a preprocessing transform to precondition
# our model. This only needs to be done for the energies, since the forces are obtained as derivatives and automatically
# capture the scale of the data. The subtraction of atomic reference energies is not necessary here, since only molecules
# of the same composition are used.
#
# For custom datasets, the data would have to be loaded via the SchNetPack `ASEAtomsData` and `AtomsDataModule` classes
# (see tutorial on [data preparation](tutorial_01_preparing_data.py)). In this case, one needs to make sure that the
# naming of properties is kept consistent with the config. The `schnetpack.properties` module provides standard names
# for a wide range of properties. Here, we use the definitions provided with the `MD17` class.
#
# In order to train force models, forces need to be included in the reference data.
# Once the dataset has been loaded, this can be checked as follows:

# %%
properties = ethanol_data.dataset[0]
print("Loaded properties:\n", *["{:s}\n".format(i) for i in properties.keys()])

# %% [markdown]
# As can be seen, `energy` and `forces` are included in the properties dictionary. To have a look at the `forces` array
# and check whether it has the expected dimensions, we can call:

# %%
print("Forces:\n", properties[MD17.forces])
print("Shape:\n", properties[MD17.forces].shape)

# %% [markdown]
# ## Building the model
#
# After having prepared the data in the above way, we can now build and train the force model.
# This is done in the same three steps as described in [QM9 tutorial](tutorial_02_qm9.py):
#
# 1. Defining input modules
# 2. Building the representation
# 3. Defining an output module
#
# For the representation we can use the same layers as in the previous tutorial:

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

# %% [markdown]
# Since we want to model forces, we need an additional output module. We will still use the Atomwise to predict the
# energy. However, since the forces should be described as the derivative of the energy, we have to indicate that the
# corresponding derviative of the model should be computed.
#
# This is done with the ``Forces`` module, which computes the negative derivative of the energy
#  (specified by the supplied ``energy_key``) with respect to the atom positions.

# %%
pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy)
pred_forces = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

# %% [markdown]
# The input, representation and output modules are then assembled to the neural network potential:

# %%
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=False),
    ],
)

# %% [markdown]
# ## Training the model
#
# Before we can train the model, the training task has to be defined, including the model, loss functions and
# optimizers. First, the outputs of the models are connected to their respective loss functions using `ModelOutput`.
# To train the model on energies and forces, we will use a combined loss function:
#
# \begin{equation}
# \mathcal{L}(E_\mathrm{ref},\mathbf{F}_\mathrm{ref},E_\mathrm{pred}, \mathbf{F}_\mathrm{pred}) = \frac{1}{n_\text{train}} \sum_{n=1}^{n_\text{train}} \left[  \rho_1 \left( E_\mathrm{ref} - E_\mathrm{pred} \right)^2 + \frac{\rho_2}{3N_\mathrm{atoms}} \sum^{N_\mathrm{atoms}}_\alpha \left\| \mathbf{F}_\mathrm{ref}^{(\alpha)} - \mathbf{F}_\mathrm{pred}^{(\alpha)} \right\|^2 \right],
# \end{equation}
#
# where we take the predicted forces to be:
#
# \begin{equation}
# \mathbf{F}_\mathrm{pred}^{(\alpha)} = -\frac{\partial E_\mathrm{pred}}{\partial \mathbf{R}^{(\alpha)}}.
# \end{equation}
#
# We have introduced the loss weights $\rho_i$ in order to control the tradeoff between energy and force loss.
# In SchNetPack, we can implement such a weighted loss function by setting the loss weights of `ModelOutput`:

# %%
output_energy = spk.task.ModelOutput(
    name=MD17.energy,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)

output_forces = spk.task.ModelOutput(
    name=MD17.forces,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)

# %% [markdown]
# Now, the training task can be assembled as in the last tutorial:
#

# %%
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
)

# %% [markdown]
# Finally, we train the model using the PyTorch Lightning `Trainer` for 5 epochs.

# %%
logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss",
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=5,  # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=ethanol_data)

# %% [markdown]
# Training will produce several files in the `model_path` directory, which is `forcetut` in our case.
# The split is stored in `split.npz`.
# Checkpoints are written to `checkpoints` periodically, which can be used to restart training.
# A copy of the best model is stored as `best_inference_model`, which can directly be accessed using the `torch.load`
# function.
#
# You can have a look at the log using Tensorboard:
# ```
# tensorboard --logdir=forcetut/lightning_logs
# ```

# %% [markdown]
# It should be noted that the model trained here is used exclusively for demonstrative purposes. Accordingly, its size
# and the training time have been reduced significantly. This puts strong constraints on the accuracy that can be
# obtained. For practical applications, one would e.g. increase the number of features, the interaction layers, the
# learning rate schedule and train until convergence (removing the `n_epochs` keyword from the `trainer`).
# To quickly get started training state-of-the-art models, have a look at the command line interface,
# which comes with a series of pre-built configurations.

# %% [markdown]
# ## Using the model
#
# Since all models in SchNetPack are stored in the same way, we can use the trained force model in exactly the same manner
# as described in the [QM9 tutorial](tutorial_02_qm9.py). To load the model stored in the `best_inference_model` file,
# we use the `torch.load` function. It will automatically be moved to the device it was trained on. The `AtomsConverter` can then be used to directly operate on ASE atoms objects (e.g. a molecule loaded from a file).

# %%
# set device
device = "cuda"

# load model
model_path = os.path.join(forcetut, "best_inference_model")
best_model = load_model(model_path, device=device)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
)

# create atoms object from dataset
structure = ethanol_data.test_dataset[0]
atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
)

# convert atoms to SchNetPack inputs and perform prediction
inputs = converter(atoms)
results = best_model(inputs)

print(results)

# %% [markdown]
# ## Interface to ASE
#
# Having access to molecular forces also makes it possible to perform a variety of different simulations.
# The `SpkCalculator` offers a simple way to perform all computations available in the ASE package ([QM9 tutorial](tutorial_02_qm9.py)).
# Below, we create an ASE calculator from the trained model and the previously generated `atoms` object
# (see [Preparing the data](#Preparing-the-data)).
# One important point is, that the MD17 dataset uses kcal/mol and kcal/mol/&#8491; as units for energies and forces.
# For use with ASE, these need to be converted to the standard internal ASE units eV and eV/&#8491;.
# To do so, we need to pass the units of the energies and positions used by the model to the calculator.
# The calculator will use these to derive the units of properties like forces and stress.

# %%
calculator = spk.interfaces.SpkCalculator(
    model=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key=MD17.energy,
    force_key=MD17.forces,
    energy_unit="kcal/mol",
    position_unit="Ang",
)

atoms.calc = calculator

print("Prediction:")
print("energy:", atoms.get_total_energy())
print("forces:", atoms.get_forces())

# %% [markdown]
# Among the simulations which can be done by using ASE and a force model are geometry optimisation,
# normal mode analysis and simple molecular dynamics simulations.
#
# The `AseInterface` of SchNetPack offers a convenient way to perform basic versions of these computations.
# Only a file specifying the geometry of the molecule and a pretrained model are needed.
#
# We will first generate a XYZ file containing an ethanol configuration:

# %%
# Generate a directory for the ASE computations
ase_dir = os.path.join(forcetut, "ase_calcs")

if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

# Write a sample molecule
molecule_path = os.path.join(ase_dir, "ethanol.xyz")
io.write(molecule_path, atoms, format="xyz")

# %% [markdown]
# The `AseInterface` is initialized by passing the path to the molecule, the model and a computation directory.
# In addition, the names of energies and forces model output,
# as well as their units, need to be provided (similar to the `SpkCalculator`).
# Computation device and floating point precision can be set via the `device` and `dtype` arguments.

# %%
ethanol_ase = spk.interfaces.AseInterface(
    molecule_path,
    ase_dir,
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key=MD17.energy,
    force_key=MD17.forces,
    energy_unit="kcal/mol",
    position_unit="Ang",
    device="cpu",
    dtype=torch.float64,
)

# %% [markdown]
# ### Geometry optimization
#
# For some applications it is neccessary to relax a molecule to an energy minimum.
# In order to perform this optimization of the molecular geometry, we can simply call

# %%
ethanol_ase.optimize(fmax=1e-2)

# %% [markdown]
# Since we trained only a reduced model, the accuracy of energies and forces is not optimal and several steps are
# needed to optimize the geometry.
#
# ### Normal mode analysis
#
# Once the geometry was optimized, normal mode frequencies can be obtained from the Hessian (matrix of second derivatives)
# of the molecule. The Hessian is a measure of the curvature of the potential energy surface and normal mode frequencies
# are useful for determining, whether an optimization has reached a minimum. Using the `AseInterface`, normal mode
# frequencies can be obtained via:

# %%
ethanol_ase.compute_normal_modes()

# %% [markdown]
# Imaginary frequencies indicate, that the geometry optimisation has not yet reached a minimum.
# The `AseInterface` also creates an `normal_modes.xyz` file which can be used to visualize the vibrations with jmol.
#
# ### Molecular dynamics
#
# Finally, it is also possible to basic run molecular dynamics simulations using this interface.
# To do so, we first need to prepare the system, where we specify the simulation file.
# This routine automatically initializes the velocities of the atoms to a random number corresponding to a certain average
# kinetic energy.

# %%
ethanol_ase.init_md("simulation")

# %% [markdown]
# The actual simulation is performed by calling the function `run_md` with a certain number of steps:

# %%
ethanol_ase.run_md(1000)

# %% [markdown]
# During simulation, energies and geometries are logged to `simulation.log` and `simulation.traj`, respectively.
#
# We can for example visualize the evolution of the systems total and potential energies as
#

# %%
# Load logged results
results = np.loadtxt(os.path.join(ase_dir, "simulation.log"), skiprows=1)

# Determine time axis
time = results[:, 0]

# Load energies
energy_tot = results[:, 1]
energy_pot = results[:, 2]
energy_kin = results[:, 3]

# Construct figure
plt.figure(figsize=(14, 6))

# Plot energies
plt.subplot(2, 1, 1)
plt.plot(time, energy_tot, label="Total energy")
plt.plot(time, energy_pot, label="Potential energy")
plt.ylabel("E [eV]")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, energy_kin, label="Kinetic energy")
plt.ylabel("E [eV]")
plt.xlabel("Time [ps]")
plt.legend()

temperature = results[:, 4]
print("Average temperature: {:10.2f} K".format(np.mean(temperature)))

plt.show()

# %% [markdown]
# As can be seen, the potential and kinetic energies fluctuate, while the total energy (sum of potential and kinetic
# energy) remains approximately constant. This is a good demonstration for the energy conservation obtained by modeling
# forces as energy derivatives. Unfortunately, this also means that energy conservation is not a sufficient measure for
# the quality of the potential.
#
# However, frequently one is interested in simulations where the system is coupled to an external heat bath.
# This is the same as saying that we wish to keep the average kinetic energy of the system and hence temperature close to a
# certain value. Currently, the average temperature only depends on the random velocities drawn during the initialization
# of the dynamics. Keeping a constant temperature average can be achieved by using a so-called thermostat.
# In the `AseInterface`, simulations with a thermostat (to be precise a Langevin thermostat) can be carried out by
# providing the `temp_bath` keyword. A simulation with e.g. the target temperature of 300K is performed via:

# %%
ethanol_ase.optimize(fmax=1e-2)  # reoptimize structure

ethanol_ase.init_md("simulation_300K", temp_bath=300, reset=True)
ethanol_ase.run_md(20000)

# %% [markdown]
# We can now once again plot total and potential energies.
# Instead of the kinetic energy, we plot the temperature (both quantities are directly related).

# %%
skip_initial = 5000

# Load logged results
results = np.loadtxt(os.path.join(ase_dir, "simulation_300K.log"), skiprows=1)

# Determine time axis
time = results[skip_initial:, 0]
# 0.02585
# Load energies
energy_tot = results[skip_initial:, 1]
energy_pot = results[skip_initial:, 2]

# Construct figure
plt.figure(figsize=(14, 6))

# Plot energies
plt.subplot(2, 1, 1)
plt.plot(time, energy_tot, label="Total energy")
plt.plot(time, energy_pot, label="Potential energy")
plt.ylabel("Energies [eV]")
plt.legend()

# Plot Temperature
temperature = results[skip_initial:, 4]

# Compute average temperature
print("Average temperature: {:10.2f} K".format(np.mean(temperature)))

plt.subplot(2, 1, 2)
plt.plot(time, temperature, label="Simulation")
plt.ylabel("Temperature [K]")
plt.xlabel("Time [ps]")
plt.plot(time, np.ones_like(temperature) * 300, label="Target")
plt.legend()
plt.show()

# %% [markdown]
# Since our molecule is now subjected to external influences via the thermostat the total energy is no longer conserved.
# However, the simulation temperature now fluctuates near to the requested 300K.
# This can also be seen by computing the temperature average over time, which is now close to the desired value in contrast to the previous simulation.

# %% [markdown]
# ## Summary
#
# In this tutorial, we have trained a SchNet model on energies and forces using the MD17 ethanol dataset as an example.
# We have then evaluated the performance of the model and performed geometry optimisation, normal mode analysis and basic molecular dynamic simulations using the SchNetPack ASE interface.
#
# While these simulations can already be useful for practical applications, SchNetPack also comes with its own molecular dynamics package.
# This package makes it possible to run efficient simulations on GPU and also offers access to advanced techniques, such as ring polymer dynamics.
# In the next tutorial, we will cover how to perform molecular dynamics simulations directly with SchNetPack.

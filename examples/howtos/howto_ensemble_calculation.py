# %% [markdown]
# # **SchNetPack Ensemble Calculator for Atomistic Simulations**

# %%
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize.lbfgs import LBFGS
from ase.visualize import view
from tqdm import tqdm

import schnetpack.transform as trn
from schnetpack.datasets import MD17
from schnetpack.interfaces.ase_interface import (
    AbsoluteUncertainty,
    RelativeUncertainty,
    SpkEnsembleCalculator,
)

# %% [markdown]
# This tutorial demonstrates how to use SchNetPack’s ensemble calculator to predict atomic energies and forces with uncertainty estimation.
#
# We’ll walk through the following examples:
# - How to calculate ensemble-based uncertainty
# - Structure relaxation using ensemble predictions and uncertainty
# - Running molecular dynamics (MD) simulations with uncertainty
#
# These tools are useful for identifying uncertain regions in simulation trajectories and making more informed decisions in atomistic modeling.
#

# %%
np.random.seed(42)

# %% [markdown]
# ## Ensemble Interface to ASE

# %% [markdown]
# We specify a list of PaiNN models trained on ethanol structures from the rMD17 dataset. These models constitute the **ensemble**, which will serve as a testbed for SchNetPack’s ensemble-enabled `SpkEnsembleCalculator`.
#
# **Note**: The models have been trained on 1000 samples only.

# %%
model_path_list = [
    "../trained_models/rmd17_ethanol/painn_1/best_model",
    "../trained_models/rmd17_ethanol/painn_2/best_model",
    "../trained_models/rmd17_ethanol/painn_3/best_model",
    "../trained_models/rmd17_ethanol/painn_4/best_model",
    "../trained_models/rmd17_ethanol/painn_5/best_model",
]

# %% [markdown]
# ### ⚖️ Creating an Ensemble Calculator with Uncertainty Quantification
#
# In this section, we instantiate two different uncertainty estimators:
# - `AbsoluteUncertainty` calculates raw standard deviation values.
# - `RelativeUncertainty` gives uncertainty as a fraction of the mean, which helps when comparing predictions on different scales.
#
# Both uncertainty methods are bundled together in `SpkEnsembleCalculator`. This lets us evaluate uncertainty in multiple ways for the same prediction run, giving a more complete picture of model confidence.
#
# Finally, we create the `SpkEnsembleCalculator`, which uses multiple trained models to make predictions. It also estimates uncertainty using the methods we provided. This calculator will act just like a regular ASE calculator but with built-in support for ensemble averaging and uncertainty tracking.
#
# Note that you can also define custom uncertainty methods and pass them to the `SpkEnsembleCalculator`.

# %%
uncertainty_abs = AbsoluteUncertainty(energy_weight=0.5, force_weight=1.0)
uncertainty_rel = RelativeUncertainty(energy_weight=1.0, force_weight=2.0)

uncertainty = [uncertainty_abs, uncertainty_rel]

ensemble_calculator = SpkEnsembleCalculator(
    models=model_path_list,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key=MD17.energy,
    force_key=MD17.forces,
    energy_unit="kcal/mol",
    position_unit="Ang",
    uncertainty_fn=uncertainty,
)

# %% [markdown]
# Assign the ensemble calculator `ensemble_calculator` to the atoms object

# %%
# load data into atoms object
atoms = read("../../tests/testdata/md_ethanol.xyz", index=0)
# specify atoms calculator
atoms.calc = ensemble_calculator

# %% [markdown]
#  🔮 Prediction Output:
# - ⚡ Energy: Total potential energy of the atomic system.
# - 🔧 Forces: Atomic forces for optimization or molecular dynamics.
# - 📊 Uncertainty: Estimation of model prediction uncertainty from the ensemble.

# %%
print("Prediction:")
print("energy:", atoms.get_total_energy())
print("forces:", atoms.get_forces())
print("uncertainty:", ensemble_calculator.get_uncertainty(atoms))

# %% [markdown]
# ## Structure Optimization

# %% [markdown]
# 🏗️ **Distort molecular structure:**
# - A small random disturbance is added to the atomic positions.
# - This simulates a noisy or slightly perturbed structure, which helps us test how sensitive the ensemble model is to input changes.
# - It also makes the uncertainty values more meaningful by introducing some variability.

# %%
# distort the structure
atoms.positions += np.random.normal(0, 0.1, atoms.positions.shape)

# %% [markdown]
# ⚙️ **Set Up Optimization:**
# - 🧑‍🔬 **Optimizer**: We set up the LBFGS optimizer, which is a gradient-based method used to minimize the energy of the atomic system.
# - 🔬 **Calculator**: Assign the ensemble calculator to the atoms object, this connects the prediction engine (our ensemble of models) to the atomic system.

# %%
optimizer = LBFGS(atoms)
atoms.calc = ensemble_calculator

# %% [markdown]
# 🔄 **Optimization Loop with Uncertainty Tracking:**
# - 🧑‍🔬 **Optimizer**: Run the optimization using the LBFGS algorithm with a force tolerance of 0.01 and a maximum of 100 steps.
# - 📊 **Uncertainty Tracking**: After each optimization step, the uncertainty of the energy prediction is appended to the `uncertainties` list, providing insight into model confidence during the process.

# %%
uncertainties = []

for _ in optimizer.irun(fmax=0.05, steps=300):
    uncertainties.append(ensemble_calculator.get_uncertainty(atoms))

# %% [markdown]
# Since we're using an ensemble of models, we can now estimate the uncertainty in our predictions during the optimization process:
# - **Absolute** and **Relative Uncertainty** values are extracted from the optimization steps.
# - Plot both **absolute** and **relative** uncertainties against the optimization steps to visualize how uncertainty changes during the process.

# %%
# Extract individual uncertainty types
abs_vals = [d["AbsoluteUncertainty"] for d in uncertainties]
rel_vals = [d["RelativeUncertainty"] for d in uncertainties]
steps = list(range(len(uncertainties)))

# Create figure and first axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot absolute uncertainty on left y-axis
ax1.plot(steps, abs_vals, label="Absolute Uncertainty", marker="o", color="tab:blue")
ax1.set_xlabel("Optimization Step")
ax1.set_ylabel("Absolute Uncertainty", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True)

# Create second y-axis for relative uncertainty
ax2 = ax1.twinx()
ax2.plot(steps, rel_vals, label="Relative Uncertainty", marker="x", color="tab:red")
ax2.set_ylabel("Relative Uncertainty", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Title and layout
plt.title("Uncertainty during Optimization")
fig.tight_layout()
plt.show()

# %% [markdown]
# While the absolute uncertainty rapidly decreases and remains consistently low and stable thereafter, the relative uncertainty increases as the structure optimization converges. This rise in relative uncertainty is due to the diminishing force magnitudes: although the prediction uncertainty stays nearly constant, the mean predicted values become very small, leading to a larger ratio between uncertainty and prediction.

# %% [markdown]
# ## Molecular Dynamics With Increasing Temperature

# %% [markdown]
# We now investigate the behavior of the uncertainty measure during a MD simulation. To this end, we perform a simulation in the canonical ensemble (NVT), gradually increasing the temperature of the heat bath throughout the run. As the temperature rises, we expect larger deviations of the molecular structure from equilibrium configurations. Consequently, the system is more likely to sample structures that lie outside the training distribution of the machine learning force field. This effect is reflected in the absolute uncertainty measure, which increases with temperature.
#
# In this setup, we use only absolute uncertainty to measure how much the model predictions vary across the ensemble:
#
# 🔎 **Note**: The `uncertainty_fn` can be passed as either a **single** uncertainty function or as a **list** of uncertainty functions.
#

# %%
uncertainty_abs = AbsoluteUncertainty(energy_weight=0.5, force_weight=1.0)

abs_ensemble_calculator = SpkEnsembleCalculator(
    models=model_path_list,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key=MD17.energy,
    force_key=MD17.forces,
    energy_unit="kcal/mol",
    position_unit="Ang",
    uncertainty_fn=uncertainty_abs,
)

# %%
target_temperatures = [_ for _ in range(50, 800, 100)]
n_steps = 1000
sampling_interval = 10
step_size = 0.5

# setting up initial atoms
atoms = read("../../tests/testdata/md_ethanol.xyz", index=0)
atoms.calc = abs_ensemble_calculator

MaxwellBoltzmannDistribution(atoms, temperature_K=target_temperatures[0])

ats_traj = []
uncertainties = []
temp = []

for target_temperature in target_temperatures:
    print(f"Temp: {target_temperature:.2f} K")
    for step in tqdm(range(n_steps // sampling_interval)):
        dyn = Langevin(
            atoms,
            timestep=step_size * units.fs,
            temperature_K=target_temperature,
            friction=0.01 / units.fs,
        )

        dyn.run(sampling_interval)

        temp.append(atoms.get_temperature())
        uncertainties.append(abs_ensemble_calculator.get_uncertainty(atoms))

        ats_traj.append(atoms.copy())

# %%
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(uncertainties, marker="o", color="blue", label="Uncertainty")
ax1.set_xlabel("MD Step")
ax1.set_ylabel("Uncertainty", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(temp, marker="x", color="red", label="Temperature")
ax2.set_ylabel("Temperature (K)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Molecular Dynamics: Uncertainty and Temperature Profile")
ax1.grid(True)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# Let's visualize the MD trajectory of the structure to make sure that nothing went wrong

# %%
view(ats_traj)

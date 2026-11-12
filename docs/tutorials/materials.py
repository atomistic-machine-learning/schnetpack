# %% [markdown]
# # Force fields for materials
#
# This page trains a force field for a crystal: bulk copper in a periodic cell. It
# shows what changes compared to molecules: the cell, periodic boundary conditions,
# neighbors in the periodic images, and the stress as a third training target. The
# trained model then gives an equation of state and relaxes a strained cell.
#
# The data is a stand-in so that the page runs in a few minutes on a CPU: copper
# crystals labelled with ASE's EMT calculator, a cheap empirical potential. The
# last section shows how to load the Materials Project data instead. The
# [quickstart](../getting_started/quickstart.py) and the
# [data tutorial](data.py) explain the steps shared with molecules.

# %%
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack.lightning import AtomisticTask, AtomsDataModule

# everything this page writes goes here; use a directory of your own instead
workdir = tempfile.mkdtemp()

# %% [markdown]
# ## Data
#
# A periodic structure is a cell, three vectors that span a box, and the atoms in
# it. Along each direction where the periodic boundary conditions (pbc) are on,
# the box repeats without end. ASE's `bulk` builds the cubic cell of copper: a
# cube with an edge of 3.59 Å that holds 4 atoms.
#
# To sample more than the perfect crystal, each of 300 structures gets a random
# strain of the cell and random displacements of the atoms. The strain has an
# isotropic part, which changes the volume by up to 12 %, and a smaller
# anisotropic part, which changes the shape. EMT labels each structure with its
# energy, the forces on the atoms, and the stress on the cell.

# %%
a0 = 3.59  # EMT lattice constant of copper, Å
crystal = bulk("Cu", "fcc", a=a0, cubic=True)

rng = np.random.default_rng(0)
atoms_list, property_list = [], []
for _ in range(300):
    strain = rng.uniform(-0.04, 0.04) * np.eye(3)  # isotropic
    shear = rng.normal(scale=0.01, size=(3, 3))
    strain += (shear + shear.T) / 2  # anisotropic, symmetric
    atoms = crystal.copy()
    atoms.set_cell(crystal.cell[:] @ (np.eye(3) + strain), scale_atoms=True)
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    atoms.calc = EMT()
    atoms_list.append(atoms)
    property_list.append(
        {
            "energy": np.array([atoms.get_potential_energy()]),
            "forces": atoms.get_forces(),
            "stress": atoms.get_stress(voigt=False)[None],  # shape (1, 3, 3)
        }
    )

# %% [markdown]
# The database keeps each `Atoms` object with its cell and pbc, so a periodic
# structure needs nothing extra. The stress is a 3×3 tensor per structure. It is
# stored with a leading axis of length one, shape `(1, 3, 3)`, so that a batch of
# structures stacks to `(n_structures, 3, 3)`. ASE gives it in eV/Å³
# (1 eV/Å³ = 160.2 GPa), positive under tension.

# %%
datapath = os.path.join(workdir, "copper_emt.db")
db = ASEAtomsData.create(
    datapath,
    distance_unit="Ang",
    property_unit_dict={"energy": "eV", "forces": "eV/Ang", "stress": "eV/Ang/Ang/Ang"},
)
db.add_systems(property_list, atoms_list)

structure = db[0]
print("cell (Å):\n", structure[properties.cell][0].numpy().round(3))
print("pbc:", structure[properties.pbc].numpy())
print("stress (GPa):\n", (structure["stress"][0] / units.GPa).numpy().round(2))

# %% [markdown]
# ## Neighbors in the periodic images
#
# The neighbor list finds, for every atom, the atoms within the cutoff. With
# periodic boundary conditions, a neighbor may lie in a copy of the cell. The list
# then stores the index `j` of the atom in the cell and an offset: the vector from
# the cell to the copy, a whole number of cell vectors. The model computes the
# distance vector of a pair `i, j` as `R[j] - R[i] + offsets`.
#
# Here the cutoff of 5 Å is longer than the cell, so each atom meets the others
# many times, in different copies of the cell. The neighbor list of the first
# structure adds the pair indices `idx_i`, `idx_j` and the `offsets` (see the
# [glossary](../concepts/glossary.md) for the key names).

# %%
cutoff = 5.0

structure = trn.MatScipyNeighborList(cutoff=cutoff)(structure)
idx_i = structure[properties.idx_i]
idx_j = structure[properties.idx_j]
offsets = structure[properties.offsets]
R = structure[properties.R]
print(f"{len(idx_i)} pairs for {len(R)} atoms")

print("neighbors of atom 0:")
for j in range(len(R)):
    pair = (idx_i == 0) & (idx_j == j)
    distances = (R[j] - R[0] + offsets[pair]).norm(dim=1)
    print(
        f"  atom {j}: {int(pair.sum()):2d} copies, "
        f"{distances.min():.2f} to {distances.max():.2f} Å"
    )

shifts = offsets @ torch.linalg.inv(structure[properties.cell][0])
own = (idx_i == 0) & (idx_j == 0)
print("cell shifts of the copies of atom 0:", shifts[own].round().int().tolist())

# %% [markdown]
# Atom 0 is its own neighbor: its copies in the six adjacent cells lie one cell
# edge away, shifted by one cell vector. Without the offsets, these pairs would
# have a distance of zero.
#
# The dataset puts the neighbor list into its `transforms`, as for molecules. The
# batch transforms remove the mean energy and cast to single precision. With
# `cache=True`, the data module reads every structure once, at the start, and keeps
# the result in memory: the neighbor lists of periodic structures are worth
# computing only once. Without it, the dataset would build them again at every
# read. The 300 structures are split into 240 for training, 30 for validation and
# 30 for testing.

# %%
dataset = ASEAtomsData(datapath, transforms=[trn.MatScipyNeighborList(cutoff=cutoff)])
datamodule = AtomsDataModule(
    dataset,
    batch_transforms=[trn.RemoveOffsets("energy", remove_mean=True), trn.CastTo32()],
    batch_size=16,
    num_train=240,
    num_val=30,
    cache=True,
    split_file=os.path.join(workdir, "split.npz"),
)

# %% [markdown]
# ## Model
#
# The stress is the derivative of the energy with respect to a strain of the
# cell, divided by the cell volume. To take this derivative, the model needs the
# strain as an input. The input module {class}`~schnetpack.model.Strain` adds a
# strain tensor of zeros and deforms the cell, the positions and the offsets with
# it. It must come before `PairwiseDistances`, so that the distances depend on
# the strain.
#
# {class}`~schnetpack.model.Forces` then computes two derivatives of the energy:
# the forces, with respect to the positions, and with `calc_stress=True` the
# stress, with respect to the strain.

# %%
n_atom_basis = 32

model = spk.model.NeuralNetworkPotential(
    representation=spk.model.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=3,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    ),
    input_modules=[spk.model.Strain(), spk.model.PairwiseDistances()],
    output_modules=[
        spk.model.Atomwise(n_in=n_atom_basis, output_key="energy"),
        spk.model.Forces(calc_forces=True, calc_stress=True),
    ],
    postprocessors=[trn.CastTo64(), trn.AddOffsets("energy", add_mean=True)],
)

# %% [markdown]
# ## Training
#
# The loss is a weighted sum of the mean squared errors of energy, forces and
# stress. The stress values are small numbers in eV/Å³, so they get a large
# weight to count about as much as the forces.

# %%
outputs = [
    spk.ModelOutput(
        name=name,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=weight,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )
    for name, weight in [("energy", 0.01), ("forces", 1.0), ("stress", 100.0)]
]
task = AtomisticTask(
    model=model,
    outputs=outputs,
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-3},
)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    # forces and stress are gradients, so validation and testing need autograd
    inference_mode=False,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    enable_model_summary=False,
)
pl.seed_everything(0)
trainer.fit(task, datamodule=datamodule)

# %% [markdown]
# The mean absolute errors on the test set:

# %%
trainer.test(task, datamodule=datamodule, verbose=False)
mae = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("test")}
print(f"energy: {mae['test_energy_MAE'] / len(crystal) * 1000:.1f} meV/atom")
print(f"forces: {mae['test_forces_MAE']:.3f} eV/Å")
print(f"stress: {mae['test_stress_MAE'] / units.GPa:.2f} GPa")

# %% [markdown]
# ## Equation of state
#
# {class}`~schnetpack.interfaces.SpkCalculator` wraps the model as an ASE
# calculator. It builds the neighbor list of each structure it is given, with the
# periodic images. `stress_key` names the stress output of the model, so that
# ASE can ask for the stress.
#
# The equation of state is the energy as a function of the volume. We compress
# and stretch the perfect crystal and compare the model with EMT. A fit gives the
# equilibrium volume and the bulk modulus.

# %%
calculator = spk.interfaces.SpkCalculator(
    model=task.model,
    neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff),
    stress_key="stress",
    energy_unit="eV",
    position_unit="Ang",
)

volumes, e_model, e_emt = [], [], []
for scale in np.linspace(0.96, 1.04, 9):
    atoms = crystal.copy()
    atoms.set_cell(crystal.cell[:] * scale, scale_atoms=True)
    volumes.append(atoms.get_volume() / len(atoms))
    atoms.calc = calculator
    e_model.append(atoms.get_potential_energy() / len(atoms))
    atoms.calc = EMT()
    e_emt.append(atoms.get_potential_energy() / len(atoms))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(volumes, e_emt, "-", label="EMT")
ax.plot(volumes, e_model, "o", label="model")
ax.set_xlabel("volume per atom (Å³)")
ax.set_ylabel("energy per atom (eV)")
ax.legend()
plt.show()

for name, energies in [("EMT", e_emt), ("model", e_model)]:
    v0, _, bulk_modulus = EquationOfState(volumes, energies).fit()
    print(f"{name:5s}  V0 = {v0:.2f} Å³/atom   B = {bulk_modulus / units.GPa:.0f} GPa")

# %% [markdown]
# ## Relaxing the cell
#
# With the stress, an optimizer can change the cell as well as the positions.
# ASE's `FrechetCellFilter` hands both to the optimizer. We take a larger cell of
# 2×2×2 cubic cells, 32 atoms, shear it and displace its atoms. The model has
# only seen cells of 4 atoms, but each atom's energy depends only on its
# neighbors, so it applies to any cell size. After relaxing with the model, EMT
# checks the result: its forces and stress should be close to zero.

# %%
atoms = crystal.repeat(2)
deformation = np.eye(3) + np.array(
    [[0.03, 0.01, 0.0], [0.01, -0.02, 0.0], [0.0, 0.0, 0.01]]
)
atoms.set_cell(atoms.cell[:] @ deformation, scale_atoms=True)
atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
atoms.calc = calculator
BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=0.01, steps=200)

print("relaxed cell lengths (Å):", atoms.cell.lengths().round(3))
print("relaxed cell angles (°): ", atoms.cell.angles().round(2))
reference = atoms.copy()
reference.calc = EMT()
print(f"largest EMT force:  {np.abs(reference.get_forces()).max():.3f} eV/Å")
print(f"largest EMT stress: {np.abs(reference.get_stress()).max() / units.GPa:.2f} GPa")

# %% [markdown]
# The cell returns to a cube with an edge of twice the lattice constant,
# 2 × 3.59 Å = 7.18 Å.
#
# ## Real data
#
# For real materials, the training data comes from density functional theory.
# {class}`~schnetpack.datasets.MaterialsProject` downloads the crystals of the
# Materials Project with their energy per atom, formation energy, band gap and
# magnetization, for property prediction. The download needs a free API key from
# the Materials Project website and the packages `mp-api` and `pymatgen`.
# {class}`~schnetpack.datasets.MPTraj` holds energies, forces and stresses of
# relaxation frames, for force fields like the one above; its download is 12 GB.
#
# The cell below shows how to load them. It is not run when this page is built;
# the page uses the small EMT data instead.

# %% tags=["skip-execution"]
from schnetpack.datasets import MaterialsProject, MPTraj

mp = MaterialsProject(
    "materials_project.db",  # downloaded on first use
    apikey="<your 32-character API key>",
    transforms=[trn.MatScipyNeighborList(cutoff=cutoff)],
)
mptraj = MPTraj(
    "mptraj.db",  # downloaded on first use
    transforms=[trn.MatScipyNeighborList(cutoff=cutoff)],
)
# offsets and the cast go to AtomsDataModule(..., batch_transforms=[...]), as above

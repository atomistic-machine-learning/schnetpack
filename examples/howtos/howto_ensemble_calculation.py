# %% [markdown]
# # Ensembles and Uncertainty

# %% [markdown]
# A single machine-learned force field gives no hint of when it is wrong. An *ensemble* does: several models trained on the same task agree where they have seen data and drift apart where they have not. The spread of their predictions is therefore a cheap estimate of the error, one number per structure, and it can flag structures that a simulation should not trust.
#
# This howto uses `EnsembleCalculator` from `schnetpack.dynamics`. It evaluates every model of the ensemble on a whole batch of structures and returns their mean, which drives any batch-wise driver (here `LBFGS`) exactly as a single model's output would. It also returns one uncertainty per structure. We will
#
# 1. build an ensemble of five PaiNN models and look at its outputs,
# 2. relax a batch of distorted ethanol structures and follow the uncertainty of every structure along the way,
# 3. see how the uncertainty grows as structures move away from the training data, and
# 4. run the same ensemble in ase, for molecular dynamics with a rising temperature.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import units
from ase.constraints import FixCom
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from ase.neighborlist import NeighborList, natural_cutoffs

import schnetpack as spk
from schnetpack.dynamics import LBFGS, BatchNeighborList, EnsembleCalculator, Hook
from schnetpack.interfaces.ase_interface import (
    SpkEnsembleCalculator,
    atoms_to_batch,
    batch_to_atoms,
)
from schnetpack.uncertainty import AbsoluteUncertainty

# the example models were saved by an older schnetpack and warn about it on loading
warnings.filterwarnings("ignore", message="Model was saved without version")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# %% [markdown]
# ## The ensemble
#
# The ensemble consists of five PaiNN models, trained on 1000 ethanol structures of the rMD17 dataset each, with different random initialisations and data splits. They were trained with schnetpack, which leaves every run in a directory of its own:
#
# ```
# trained_models/rmd17_ethanol/
# ├── painn_1/best_model
# ├── ...
# └── painn_5/best_model
# ```
#
# `EnsembleCalculator` takes such a directory as it is (or a list of model paths, or of loaded models). The models run as one `NNEnsemble` on the same batch, so they share the neighbor list and the units: here a cutoff of 5 Å, and energies in kcal/mol. The calculator converts what they return, so energies come out in eV and forces in eV/Å, as for a single-model `ForceCalculator`.
#
# ### Measuring the disagreement
#
# How far the models disagree is turned into one number per structure by an `Uncertainty` from `schnetpack.uncertainty`. `AbsoluteUncertainty` takes the standard deviation $\sigma_{i\alpha}$ of every force component over the $M$ models, and averages its length over the $N$ atoms of a structure:
#
# $$u_F = \frac{1}{N} \sum_{i=1}^{N} \sqrt{\sum_{\alpha \in \{x,y,z\}} \sigma_{i\alpha}^2}\,, \qquad \sigma_{i\alpha}^2 = \frac{1}{M} \sum_{m=1}^{M} \left(F^{(m)}_{i\alpha} - \bar F_{i\alpha}\right)^2 .$$
#
# It is in eV/Å, like the forces. With `energy_weight` it measures the spread of the energy instead (in eV), and both weights together give a weighted sum of the two. `RelativeUncertainty` divides each spread by the length of the mean, which makes it dimensionless.
#
# The calculator reports a single uncertainty under the key `"uncertainty"`. A mapping reports several, each under its own key. Here we ask for the force and the energy uncertainty:

# %%
model_dir = "../trained_models/rmd17_ethanol"
cutoff = 5.0  # Angstrom, the cutoff the models were trained with

calculator = EnsembleCalculator(
    model_dir,
    energy_unit="kcal/mol",
    neighbor_list=BatchNeighborList(spk.transform.MatScipyNeighborList(cutoff=cutoff)),
    uncertainty={
        "force_uncertainty": AbsoluteUncertainty(),
        "energy_uncertainty": AbsoluteUncertainty(energy_weight=1.0, force_weight=0.0),
    },
    device=device,
    dtype=torch.float32,
)
print(f"{len(calculator.model.models)} models")

# %% [markdown]
# The structures are the three conformers of ethanol: *anti* and the two mirror-image *gauche* conformers. A batch of them goes through the ensemble in one call:

# %%
conformers = read("../../tests/testdata/ethanol_conformers.xyz", index=":")
batch = atoms_to_batch(conformers, device=device)

outputs = calculator(batch)
for key, value in outputs.items():
    print(f"{key:>20}  {tuple(value.shape)}")

# %% [markdown]
# ## Uncertainty along a batch-wise relaxation
#
# We distort five copies of each conformer with Gaussian noise of 0.1 Å on every coordinate and relax all 15 structures together with `LBFGS`. The optimizer steps on the ensemble's mean force and stops once no atomic force exceeds `fmax`.
#
# The drivers return only the final batch. To follow the uncertainty along the way we write a small `Hook`: its `after_step` evaluates the calculator on the batch the step produced and records the uncertainty of every structure. This costs no extra model call. The optimizer turns on the calculator's `cache_last`, so the stop test at the start of the next step reuses this very evaluation.

# %%
NOISE = 0.1  # Angstrom
FMAX = 0.01  # eV/Angstrom
MAX_STEPS = 500


def distort(structures, noise, n_copies, rng):
    # n_copies of every structure, each with Gaussian noise on its positions
    distorted = []
    for atoms in structures:
        for _ in range(n_copies):
            copy = atoms.copy()
            copy.positions += rng.normal(scale=noise, size=copy.positions.shape)
            distorted.append(copy)
    return distorted


class RecordUncertainty(Hook):
    # records calculator(batch)[key] after every step, one value per structure

    def __init__(self, calculator, key):
        self.calculator = calculator
        self.key = key
        self.history = []

    def after_step(self, batch, step, n_steps):
        uncertainty = self.calculator(batch)[self.key]
        self.history.append(uncertainty.detach().cpu().numpy())
        return batch


rng = np.random.default_rng(SEED)
start = distort(conformers, NOISE, n_copies=5, rng=rng)
start_batch = atoms_to_batch(start, device=device)

recorder = RecordUncertainty(calculator, "force_uncertainty")
relaxer = LBFGS(calculator, fmax=FMAX, device=device, hooks=[recorder])
relaxed_batch = relaxer.run(start_batch, MAX_STEPS)

initial = calculator(start_batch)
final = calculator(relaxed_batch)
history = np.stack([initial["force_uncertainty"].cpu().numpy()] + recorder.history)
print(f"converged after {len(recorder.history)} steps")

# %% [markdown]
# A converged structure is a minimum of the *mean* force field, but that does not make it a structure the ensemble knows. We flag every structure whose final force uncertainty is more than five times the median of the batch:

# %%
u_start = initial["force_uncertainty"].cpu().numpy()
u_end = final["force_uncertainty"].cpu().numpy()
energies = final["energy"].detach().cpu().numpy()
flagged = u_end > 5 * np.median(u_end)

print(
    f"{'structure':>9} {'conformer':>9} {'u_F start':>10} {'u_F end':>9} {'E - E_min [eV]':>15}"
)
for i in range(len(start)):
    conformer = i // 5
    e_min = energies[conformer * 5 : (conformer + 1) * 5].min()
    mark = "  <- flagged" if flagged[i] else ""
    print(
        f"{i:>9} {conformer:>9} {u_start[i]:>10.4f} {u_end[i]:>9.4f} "
        f"{energies[i] - e_min:>15.3f}{mark}"
    )

# %%
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def style(ax):
    # recessive grid and axes, text in ink rather than series colours
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
fig.patch.set_facecolor(SURFACE)
steps = np.arange(len(history))
for i in np.where(~flagged)[0]:
    ax.plot(steps, history[:, i], color=MUTED, linewidth=1.2, alpha=0.7)
for i in np.where(flagged)[0]:
    ax.plot(steps, history[:, i], color=SERIES_1, linewidth=2.2)
    ax.annotate(
        f"structure {i}",
        (steps[-1], history[-1, i]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=INK,
    )
ax.plot([], [], color=MUTED, linewidth=1.2, label="unflagged structures")
ax.plot(
    [], [], color=SERIES_1, linewidth=2.2, label="flagged (> 5 × median at the end)"
)
ax.set_yscale("log")
ax.set_xlabel("optimizer step")
ax.set_ylabel("force uncertainty $u_F$ [eV/Å]")
ax.set_title(
    "Force uncertainty of every structure during the relaxation", color=INK, loc="left"
)
ax.legend(frameon=False, labelcolor=INK_SECONDARY)
style(ax)
fig.tight_layout()
plt.show()


# %% [markdown]
# For most structures the uncertainty falls by two orders of magnitude while they relax: the noise pushed them away from the training data, and the relaxation brings them back to the minima the models know well. The final values are a few meV/Å, those of the undistorted conformers above.
#
# A flagged structure is a different matter. Its uncertainty stays high although its forces converged, so the optimizer found a minimum of the mean force field the models do not agree on. That calls for a closer look. The covalent bonds, with the cutoffs of `ase.neighborlist.natural_cutoffs`, of a flagged structure against those of the conformer it started from:


# %%
def bonds(atoms):
    # the covalent bonds, as pairs of atom symbols and indices
    neighbors = NeighborList(
        natural_cutoffs(atoms), self_interaction=False, bothways=False
    )
    neighbors.update(atoms)
    symbols = atoms.get_chemical_symbols()
    return {
        (f"{symbols[i]}{i}", f"{symbols[j]}{j}")
        for i in range(len(atoms))
        for j in neighbors.get_neighbors(i)[0]
    }


relaxed = batch_to_atoms(relaxed_batch)
for i in np.where(flagged)[0]:
    reference = bonds(conformers[i // 5])
    found = bonds(relaxed[i])
    print(
        f"structure {i}: broken {sorted(reference - found)}, new {sorted(found - reference)}"
    )

# %% [markdown]
# Here the flagged structure lost a C–H bond: a methyl hydrogen left its carbon during the relaxation, and the structure came to rest in a minimum that no training structure is close to. Its forces are converged, so the optimizer alone gives no sign of trouble. Its energy lies well above that of the other relaxed copies of the same conformer. The uncertainty catches it without any knowledge of chemistry.
#
# Which structures end up there depends on the noise drawn. With another seed, a run may have no flagged structure at all, or several.

# %% [markdown]
# ## Uncertainty and the distance from the training data
#
# How does the uncertainty depend on how far a structure lies from what the models were trained on? We distort the *anti* conformer with noise of increasing amplitude, 16 copies per amplitude, and evaluate the copies of each amplitude in one call.

# %%
amplitudes = np.array([0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3])  # Angstrom
n_copies = 16

rng = np.random.default_rng(SEED)
u_sweep = []
for amplitude in amplitudes:
    copies = distort(conformers[:1], amplitude, n_copies, rng)
    outputs = calculator(atoms_to_batch(copies, device=device))
    u_sweep.append(outputs["force_uncertainty"].cpu().numpy())
u_sweep = np.stack(u_sweep)  # (amplitude, copy)

# %%
Y_MAX = 1e2  # eV/Angstrom; copies above are drawn as triangles on the top edge

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
fig.patch.set_facecolor(SURFACE)
x = amplitudes[:, None] + np.random.default_rng(1).uniform(
    -0.004, 0.004, size=u_sweep.shape
)
on_scale = u_sweep <= Y_MAX
ax.scatter(
    x[on_scale],
    u_sweep[on_scale],
    s=36,
    color=MUTED,
    alpha=0.6,
    edgecolors=SURFACE,
    linewidths=1.0,
    label="one distorted copy",
)
ax.scatter(
    x[~on_scale],
    np.full((~on_scale).sum(), Y_MAX),
    s=64,
    marker="^",
    color=MUTED,
    edgecolors=SURFACE,
    linewidths=1.0,
    clip_on=False,
    label=f"off scale (> {Y_MAX:g} eV/Å)",
)
ax.plot(
    amplitudes,
    np.median(u_sweep, axis=1),
    color=SERIES_1,
    linewidth=2,
    marker="o",
    markersize=7,
    markeredgecolor=SURFACE,
    label="median over the copies",
)
ax.set_yscale("log")
ax.set_ylim(top=Y_MAX)
ax.set_xlabel("noise amplitude [Å]")
ax.set_ylabel("force uncertainty $u_F$ [eV/Å]")
ax.set_title("Force uncertainty of the distorted anti conformer", color=INK, loc="left")
ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
style(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# Up to about 0.05 Å, distortions of the size of the thermal fluctuations in the training data, the median uncertainty stays within a factor of a few of the undistorted structure's. Between 0.05 and 0.1 Å it rises by more than an order of magnitude, and beyond that the copies scatter over several orders of magnitude.

# %% [markdown]
# ## The same ensemble in ase
#
# `SpkEnsembleCalculator` is the ase counterpart: an ase calculator for one structure at a time, which plugs into ase's optimizers and molecular dynamics. It runs the same `NNEnsemble` and the same `Uncertainty` classes on predictions converted to eV and Å, so it reports the same numbers as the batch-wise calculator. Its `results` hold each uncertainty as a float under its key, and `get_uncertainty(atoms, key)` returns one of them.

# %%
model_paths = [f"{model_dir}/painn_{i}/best_model" for i in range(1, 6)]

ase_calculator = SpkEnsembleCalculator(
    model_paths,
    neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff),
    energy_unit="kcal/mol",
    uncertainty={"force_uncertainty": AbsoluteUncertainty()},
    device=device,
)

atoms = start[0].copy()
atoms.calc = ase_calculator
batchwise = calculator(atoms_to_batch([atoms], device=device))

print(
    f"energy       ase {atoms.get_potential_energy():.5f} eV, batch-wise {batchwise['energy'].item():.5f} eV"
)
u_ase = ase_calculator.get_uncertainty(atoms, "force_uncertainty")
u_batch = batchwise["force_uncertainty"].item()
print(f"uncertainty  ase {u_ase:.6f} eV/Å, batch-wise {u_batch:.6f} eV/Å")
assert np.isclose(u_ase, u_batch, rtol=1e-4)

# %% [markdown]
# ### Molecular dynamics with a rising temperature
#
# We run Langevin dynamics of ethanol in the canonical ensemble and raise the temperature of the heat bath in steps from 100 K to 1100 K. The hotter the bath, the further the molecule strays from the equilibrium structures the training data covers, and the larger the uncertainty should become. Every 10 steps we record the temperature and the force uncertainty.

# %%
bath_temperatures = [100, 300, 500, 700, 900, 1100]  # K
steps_per_temperature = 400
interval = 10
timestep = 0.5  # fs

atoms = read("../../tests/testdata/md_ethanol.xyz", index=0)
atoms.calc = ase_calculator
atoms.set_constraint(FixCom())  # keep the molecule from drifting
thermalize_momenta(
    atoms, temperature_K=bath_temperatures[0], rng=np.random.default_rng(SEED)
)
dynamics = Langevin(
    atoms,
    timestep=timestep * units.fs,
    temperature_K=bath_temperatures[0],
    friction=0.01 / units.fs,
    fixcm=False,
    rng=np.random.default_rng(SEED),
)
bath = {"temperature": bath_temperatures[0]}

record = {"time": [], "temperature": [], "uncertainty": [], "bath": []}


def log_step():
    record["time"].append(dynamics.get_time() / (1000 * units.fs))  # ps
    record["temperature"].append(atoms.get_temperature())
    record["uncertainty"].append(
        ase_calculator.get_uncertainty(atoms, "force_uncertainty")
    )
    record["bath"].append(bath["temperature"])


dynamics.attach(log_step, interval=interval)
for temperature in bath_temperatures:
    bath["temperature"] = temperature
    dynamics.set_temperature(temperature_K=temperature)
    dynamics.run(steps_per_temperature)
record = {key: np.array(value) for key, value in record.items()}

# %% [markdown]
# Temperature and uncertainty have different units, so they get a panel each, on a shared time axis:

# %%
fig, (ax_t, ax_u) = plt.subplots(2, 1, figsize=(7.2, 6.0), dpi=130, sharex=True)
fig.patch.set_facecolor(SURFACE)

ax_t.plot(
    record["time"],
    record["temperature"],
    color=SERIES_2,
    linewidth=1.2,
    label="molecule",
)
ax_t.step(
    record["time"],
    record["bath"],
    where="post",
    color=INK_SECONDARY,
    linewidth=1.5,
    linestyle="--",
    label="heat bath",
)
ax_t.set_ylabel("temperature [K]")
ax_t.set_title(
    "Langevin dynamics of ethanol with a rising bath temperature", color=INK, loc="left"
)
ax_t.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
style(ax_t)

ax_u.plot(record["time"], record["uncertainty"], color=SERIES_1, linewidth=1.2)
ax_u.set_yscale("log")
ax_u.set_xlabel("time [ps]")
ax_u.set_ylabel("force uncertainty $u_F$ [eV/Å]")
style(ax_u)

fig.tight_layout()
plt.show()

# %% [markdown]
# The uncertainty follows the temperature. At low temperature the molecule vibrates close to its equilibrium structure, and the models agree. Each increase of the bath temperature lets it explore larger distortions, and the uncertainty rises with it, with spikes where a fluctuation briefly takes the molecule far from the training data. In a production run, a threshold on this signal tells when a trajectory has left the region in which the force field can be trusted: when to stop, or which structures to recompute with a reference method and add to the training data.

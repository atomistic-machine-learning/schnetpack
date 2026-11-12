# %% [markdown]
# # Molecular dynamics in SchNetPack
#
# The [force field tutorial](force_fields.py) trained a model on forces and ran
# molecular dynamics (MD) with it through ASE. SchNetPack also has an MD engine of
# its own, `schnetpack.md` ([API](../api/md.rst)). It keeps the whole simulation in
# PyTorch tensors, so it
#
# - runs on a GPU and propagates many molecules and replicas in one model call,
# - adds sampling methods such as thermostats, barostats and ring-polymer MD,
# - is built from small modules that are easy to swap or extend.
#
# This page sets up a simulation of ethanol piece by piece, adds a thermostat,
# logging and checkpoints, reads the results, and then runs ring-polymer MD and
# MD through ASE. The [MD command line guide](../guides/md_cli.md) runs the same
# simulations from config files with `spkmd`.
#
# To relax many structures at once rather than run MD, `schnetpack.dynamics` has
# batched optimizers, shown in the
# [batch-wise relaxation guide](../guides/batchwise_relaxations.py).

# %%
import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read, write

import schnetpack as spk
from schnetpack import properties
from schnetpack import units as spk_units
from schnetpack.utils import load_model

# the example model was saved by an older SchNetPack and warns about it on loading
warnings.filterwarnings("ignore", message="Model was saved without version")

torch.manual_seed(0)

# everything this page writes goes here; use a directory of your own instead
workdir = tempfile.mkdtemp()

# %% [markdown]
# ## Model and structure
#
# No training on this page: it uses a converged PaiNN model of ethanol and an
# ethanol structure, both shipped with SchNetPack's tests. The model predicts
# energies in kcal/mol from positions in Å. Its cutoff sets the neighbor list we
# need below.

# %%
model_path = "../../tests/testdata/md_ethanol.model"
molecule_path = "../../tests/testdata/md_ethanol.xyz"

model = load_model(model_path)
cutoff = model.representation.cutoff.item()  # Angstrom
print(f"{type(model.representation).__name__}, cutoff {cutoff} Å")

molecule = read(molecule_path)
print(molecule.get_chemical_formula())

# %% [markdown]
# ## How the MD code is organised
#
# An MD code keeps track of the positions $\mathbf{R}$ and momenta $\mathbf{p}$ of
# all nuclei. In every step it computes the forces $\mathbf{F}$ and uses them to
# integrate Newton's equations of motion. SchNetPack splits these jobs into
# modules:
#
# ```{image} figures/md_flowchart.svg
# :alt: The Simulator calls the Integrator, the Calculator and the System in turn.
# :width: 200px
# :align: center
# ```
#
# - The {class}`~schnetpack.md.System` holds the state: positions, momenta, atom
#   types, cells and the properties the model computed.
# - The integrator moves positions and momenta forward by one time step.
# - The calculator computes the forces for the current positions, here with the
#   trained model.
# - The {class}`~schnetpack.md.Simulator` holds the other three and runs the MD
#   loop, calling them in the right order.
#
# ### System
#
# The `System` stores everything in tensors of shape
# $N_\mathrm{replicas} \times N_\mathrm{atoms} \times \ldots$. The first dimension
# counts copies of the same molecule, used by ring-polymer MD below; for plain MD it
# is 1. The second runs over the atoms of all molecules, so molecules of different
# size need no padding.
#
# `load_molecules` takes one ASE `Atoms` object or a list of them, the number of
# replicas, and the length unit of the input. The MD code works in its own units
# internally (kJ/mol, nm, Dalton) and converts the inputs.

# %%
from schnetpack.md import System, UniformInit

system = System()
system.load_molecules(molecule, n_replicas=1, position_unit_input="Angstrom")

# %% [markdown]
# All momenta are zero after loading. An initializer draws them for a given
# temperature in K. `UniformInit` draws random momenta and rescales them to the
# temperature; `MaxwellBoltzmannInit` draws them from the Maxwell–Boltzmann
# distribution. Both can also move the molecule to its center of mass and remove
# overall translation and rotation.

# %%
temperature = 300  # K

initializer = UniformInit(
    temperature,
    remove_center_of_mass=True,
    remove_translation=True,
    remove_rotation=True,
)
initializer.initialize_system(system)

# %% [markdown]
# ### Integrator
#
# The integrator propagates the equations of motion with a fixed time step, given
# in fs. {class}`~schnetpack.md.integrators.VelocityVerlet` is the standard choice
# for classical MD; 0.5 fs is
# small enough for the fast O–H and C–H vibrations of ethanol.

# %%
from schnetpack.md.integrators import VelocityVerlet

time_step = 0.5  # fs
integrator = VelocityVerlet(time_step)

# %% [markdown]
# ### Calculator
#
# The calculator connects the model to the MD code. It needs a neighbor list first.
# {class}`~schnetpack.md.neighborlist_md.NeighborListMD` wraps any SchNetPack
# neighbor list transform for MD. It adds a
# buffer shell to the cutoff, so the list is only rebuilt after an atom has moved
# far enough to cross it. Both lengths are in the units of the model, here Å.

# %%
from schnetpack.md.neighborlist_md import NeighborListMD

neighbor_list = NeighborListMD(
    cutoff=cutoff,
    cutoff_shell=2.0,  # Angstrom
    base_nbl=spk.transform.MatScipyNeighborList,
)

# %% [markdown]
# {class}`~schnetpack.md.calculators.SchNetPackCalculator` loads the model from its
# file. It needs the name of the
# force output and the units the model works in: energies in kcal/mol and
# positions in Å (write `"Angstrom"` or `"Ang"`; `"A"` would mean ampere). It
# derives the force unit from these two. With `energy_key`, it also stores the
# potential energy in the `System`. Further outputs of the model can be requested
# with `required_properties`.
#
# The calculator switches off the model's postprocessing, so energies come out
# without the constant mean energy the model otherwise adds back. Energy
# differences are unaffected.

# %%
from schnetpack.md.calculators import SchNetPackCalculator

calculator = SchNetPackCalculator(
    model_path,
    force_key="forces",
    energy_unit="kcal/mol",
    position_unit="Angstrom",
    neighbor_list=neighbor_list,
    energy_key="energy",
)

# %% [markdown]
# ### Simulator
#
# The `Simulator` brings the three together. Like any PyTorch module it moves to a
# device and a precision with `.to`; single precision is enough for MD with this
# model. `progress=False` turns off the progress bar.

# %%
from schnetpack.md import Simulator

device = "cuda" if torch.cuda.is_available() else "cpu"

simulator = Simulator(system, integrator, calculator, progress=False)
simulator = simulator.to(torch.float32).to(device)

# %% [markdown]
# `simulate` runs a number of steps. The simulator keeps its state, so a second
# call continues the trajectory. 100 steps of 0.5 fs are 50 fs.

# %%
simulator.simulate(100)
simulator.simulate(100)
print("steps:", simulator.step)
print(f"temperature: {simulator.system.temperature.item():.0f} K")

# %% [markdown]
# This run kept no record of what happened. Collecting data, like most other
# additions to plain MD, is the job of simulation hooks.
#
# ## Simulation hooks
#
# A hook is a set of instructions the simulator runs at fixed points of every step:
#
# ```{image} figures/integrator.svg
# :alt: One MD step, with the points where hooks are called.
# :width: 370px
# :align: center
# ```
#
# - Hooks around the two half steps of the integrator control the temperature
#   (thermostats) or the pressure (barostats).
# - Hooks right after the force computation can change forces and energies, for
#   enhanced sampling such as metadynamics.
# - Hooks at the end of the step collect and store data.
#
# A simulator takes any number of hooks. We add a thermostat, a log file,
# checkpoints and a TensorBoard log.
#
# ### Thermostat
#
# A thermostat keeps the temperature close to that of a heat bath. The number of
# particles $N$, the volume $V$ and the mean temperature $T$ stay constant, so
# this is called an $NVT$ or canonical simulation. `LangevinThermostat` takes the
# bath temperature in K and a time constant in fs, which sets how fast it pulls
# the system towards the bath temperature. 100 fs is short, which speeds up
# equilibration. `schnetpack.md.simulation_hooks` has more thermostats, e.g.
# Berendsen, Nosé–Hoover chains and GLE.

# %%
from schnetpack.md.simulation_hooks import LangevinThermostat

thermostat = LangevinThermostat(temperature, time_constant=100)  # K, fs

# %% [markdown]
# ### Logging to an HDF5 file
#
# {class}`~schnetpack.md.simulation_hooks.FileLogger` writes the trajectory to an
# HDF5 file. What it writes is set by
# data streams:
#
# - `MoleculeStream` stores the positions and, optionally, the velocities.
# - `PropertyStream` stores properties computed by the calculator, here the
#   energy, in the calculator's units.
#
# The logger collects `buffer_size` steps in memory before writing them, to save
# disk access. `every_n_steps` sets how often it records.

# %%
from schnetpack.md.simulation_hooks import FileLogger, MoleculeStream, PropertyStream

log_file = os.path.join(workdir, "simulation.hdf5")

file_logger = FileLogger(
    log_file,
    buffer_size=100,
    data_streams=[
        MoleculeStream(store_velocities=True),
        PropertyStream(target_properties=[properties.energy]),
    ],
    every_n_steps=1,
    precision=32,  # store floats in single precision
)

# %% [markdown]
# ### Checkpoints
#
# A checkpoint stores the full state of the simulation, so it can be continued
# after a crash or used as the start of a new run. `Checkpoint` writes one every
# `every_n_steps` steps and at the end of the run.

# %%
from schnetpack.md.simulation_hooks import Checkpoint

checkpoint_file = os.path.join(workdir, "simulation.chk")
checkpoint = Checkpoint(checkpoint_file, every_n_steps=100)

# %% [markdown]
# ### TensorBoard
#
# `TensorBoardLogger` logs scalars to watch a running simulation: `"energy"`
# (potential and kinetic), `"temperature"`, and for constant-pressure runs
# `"pressure"` and `"volume"`. View them with `tensorboard --logdir=<log directory>`.

# %%
from schnetpack.md.simulation_hooks import TensorBoardLogger

tensorboard_logger = TensorBoardLogger(
    os.path.join(workdir, "logs"), ["energy", "temperature"], every_n_steps=100
)

# %% [markdown]
# ### Running with hooks
#
# The hooks go into a new simulator, which continues from the current state of the
# `System`. 2000 steps are 1 ps.

# %%
hooks = [thermostat, file_logger, checkpoint, tensorboard_logger]

simulator = Simulator(
    system, integrator, calculator, simulator_hooks=hooks, progress=False
)
simulator = simulator.to(torch.float32).to(device)
simulator.simulate(2000)

print(sorted(os.listdir(workdir)))

# %% [markdown]
# ## Restarting a simulation
#
# A checkpoint is a dictionary, read with `torch.load`. `restart_simulation`
# restores the step count, the system and the thermostat state on a simulator. The
# `System` can be empty: the checkpoint fills it. The file logger sees the restart
# and appends to the existing log, which now holds 2200 steps.

# %%
state = torch.load(checkpoint_file, weights_only=False)

restarted = Simulator(
    System(),
    integrator,
    calculator,
    simulator_hooks=[
        LangevinThermostat(temperature, time_constant=100),
        FileLogger(
            log_file,
            buffer_size=100,
            data_streams=[
                MoleculeStream(store_velocities=True),
                PropertyStream(target_properties=[properties.energy]),
            ],
        ),
    ],
    progress=False,
)
restarted.restart_simulation(state)
restarted = restarted.to(torch.float32).to(device)
print("restarting at step", restarted.step)

restarted.simulate(200)

# %% [markdown]
# To take only positions and momenta, e.g. to start a production run from an
# equilibrated structure with a different thermostat, load the system state alone
# with `system.load_system_state(state["system"])`.

# %% [markdown]
# ## Reading the HDF5 log
#
# {class}`~schnetpack.md.data.HDF5Loader` reads the log file. Its `properties`
# holds everything the file contains.

# %%
from schnetpack.md.data import HDF5Loader

data = HDF5Loader(log_file)
print(list(data.properties))
print("entries:", data.entries)

# %% [markdown]
# Most keys are the standard SchNetPack
# [batch keys](../concepts/glossary.md#batch-keys). The energy
# is there twice. `energy` comes from the `PropertyStream` and is in the
# calculator's units, kcal/mol. `energy_system` is the potential energy stored in
# the `System` because we set `energy_key`, and is in the MD's internal unit,
# kJ/mol.
#
# `get_property` reads any of them. It needs to know whether the property is per
# atom (`atomistic=True`, e.g. forces) or per molecule (e.g. energies). Helpers such
# as `get_potential_energy`, `get_kinetic_energy` and `get_temperature` compute
# common quantities. Converted to the same unit, both energies agree:

# %%
energy = data.get_property(properties.energy, atomistic=False)  # kcal/mol
energy_system = data.get_potential_energy() * spk_units.convert_units(
    "kJ/mol", "kcal/mol"
)
print("energies agree:", np.allclose(energy, energy_system, atol=1e-3))

# %% [markdown]
# The temperature of the run, with the bath temperature as a dotted line. A single
# molecule of 9 atoms has few degrees of freedom, so its temperature fluctuates
# strongly, and its running mean approaches 300 K only slowly.

# %%
temperature_trace = data.get_temperature()  # K
running_mean = np.cumsum(temperature_trace) / np.arange(1, data.entries + 1)
time = np.arange(data.entries) * data.time_step / spk_units.fs

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(time, temperature_trace, lw=0.8, label="instantaneous")
ax.plot(time, running_mean, label="running mean")
ax.axhline(temperature, color="k", ls=":", lw=1)
ax.set_xlabel("time (fs)")
ax.set_ylabel("temperature (K)")
ax.legend()
plt.show()

# %% [markdown]
# The first part of a trajectory is equilibration, and averages should leave it
# out. `skip_initial` drops that many entries when loading. How long equilibration
# takes depends on the system and the thermostat; here we drop the first 500 steps
# (250 fs). Over a run this short, the mean temperature still scatters by a few tens
# of kelvin around 300 K.

# %%
equilibrated = HDF5Loader(log_file, skip_initial=500)
print("entries:", equilibrated.entries)
print(f"mean temperature: {equilibrated.get_temperature().mean():.0f} K")

# %% [markdown]
# A note on units: positions, velocities and kinetic energies in the file are in
# the internal MD units (kJ/mol, nm, Dalton, ps), temperatures in K. Properties from
# the calculator stay in the model's units unless the calculator was given a
# conversion, so the energies here are in kcal/mol.
#
# ### Structures
#
# `convert_to_atoms` returns the trajectory as a list of ASE `Atoms` in Å, which
# ASE can write to any of its file formats:

# %%
trajectory = data.convert_to_atoms()
write(os.path.join(workdir, "trajectory.xyz"), trajectory)
print(len(trajectory), "structures")

# %% [markdown]
# ### Vibrational spectrum
#
# `schnetpack.md.data` computes vibrational spectra from a log file:
# `PowerSpectrum` from the velocities, and `IRSpectrum` and `RamanSpectrum` for
# models that also predict dipole moments or polarizabilities. A power spectrum is
# the vibrational density of states. It shows every mode that *can* appear in an
# IR or Raman spectrum, but not which ones do, or how strongly.
#
# `resolution` is the number of time lags of the velocity autocorrelation. More
# lags give sharper peaks but need a longer trajectory. Ours is under 1 ps, so the
# peaks are broad.

# %%
from schnetpack.md.data import PowerSpectrum

spectrum = PowerSpectrum(equilibrated, resolution=512)
spectrum.compute_spectrum(molecule_idx=0)
frequencies, intensities = spectrum.get_spectrum()

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(frequencies, intensities)
ax.set_xlim(0, 4000)
ax.set_ylim(0)
ax.set_xlabel("wavenumber (cm$^{-1}$)")
ax.set_ylabel("intensity (a.u.)")
plt.show()

# %% [markdown]
# The bands are those expected for ethanol (compare an
# [IR absorption table](https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Spectroscopic_Parameters/Infrared_Spectroscopy_Absorption_Table)):
# the O–H stretch near 3700 cm⁻¹, the C–H stretches around 3000 cm⁻¹ and the
# bending modes around 1400 cm⁻¹. A spectrum like this is a quick check of a
# machine-learned potential.
#
# ## Ring-polymer MD
#
# Classical MD treats the nuclei as point particles and misses nuclear quantum
# effects such as zero-point energy. Ring-polymer MD (RPMD) recovers part of them.
# Each atom is replaced by several copies, called beads, that are joined by
# harmonic springs into a ring. This is a discretised path integral: the more
# beads, the closer the result to quantum mechanics.
#
# RPMD needs a model evaluation for every bead in every step, which makes cheap
# machine-learned potentials a good fit. The beads are the replica dimension of the
# `System`, so one model call computes all of them. Compared to classical MD, three
# things change:
#
# - the `System` gets one replica per bead,
# - the `RingPolymer` integrator replaces `VelocityVerlet`,
# - the thermostat must be one for ring polymers.
#
# We use 4 beads to keep the run short; converged results need more. The
# initializer from above handles the replicas.

# %%
n_beads = 4

rpmd_system = System()
rpmd_system.load_molecules(molecule, n_replicas=n_beads, position_unit_input="Angstrom")
initializer.initialize_system(rpmd_system)

# %% [markdown]
# The {class}`~schnetpack.md.integrators.RingPolymer` integrator needs the number
# of beads and the temperature of the ring polymer, which sets the stiffness of the
# springs; usually this is the bath temperature. RPMD needs a smaller time step than
# classical MD to stay stable.
#
# {class}`~schnetpack.md.simulation_hooks.PILELocalThermostat` is the ring-polymer
# counterpart of the Langevin thermostat.
# Other ring-polymer thermostats are listed in the [MD API](../api/md.rst).

# %%
from schnetpack.md.integrators import RingPolymer
from schnetpack.md.simulation_hooks import PILELocalThermostat

rpmd_time_step = 0.25  # fs
rpmd_integrator = RingPolymer(rpmd_time_step, n_beads, temperature)

rpmd_log_file = os.path.join(workdir, "rpmd.hdf5")
rpmd_hooks = [
    PILELocalThermostat(temperature, time_constant=100),
    FileLogger(
        rpmd_log_file,
        buffer_size=100,
        data_streams=[
            MoleculeStream(store_velocities=True),
            PropertyStream(target_properties=[properties.energy]),
        ],
    ),
]

# %% [markdown]
# The calculator and the simulator work as before. 800 steps of 0.25 fs are
# 200 fs.

# %%
rpmd_simulator = Simulator(
    rpmd_system,
    rpmd_integrator,
    calculator,
    simulator_hooks=rpmd_hooks,
    progress=False,
)
rpmd_simulator = rpmd_simulator.to(torch.float32).to(device)
rpmd_simulator.simulate(800)

# %% [markdown]
# For a ring polymer, `HDF5Loader` returns the centroid, the mean over all beads,
# unless `replica_idx` selects a bead. We skip the first 50 fs.
#
# Two quantum effects are visible even with 4 beads. The mean potential energy is
# higher than in classical MD at the same temperature: this is part of the
# zero-point energy. And the beads of an atom spread out around their centroid,
# the lighter the atom the more: hydrogen is far more delocalised than carbon or
# oxygen.

# %%
rpmd_data = HDF5Loader(rpmd_log_file, skip_initial=200)

# potential energy of every bead, in kcal/mol, averaged over beads and steps
rpmd_energy = np.mean(
    [
        rpmd_data.get_property(properties.energy, atomistic=False, replica_idx=k)
        for k in range(n_beads)
    ]
)
classical_energy = equilibrated.get_property(properties.energy, atomistic=False)
print(f"mean potential energy, classical MD: {classical_energy.mean():6.1f} kcal/mol")
print(f"mean potential energy, RPMD:         {rpmd_energy:6.1f} kcal/mol")

# distance of every bead from its centroid, in Angstrom
beads = np.stack([rpmd_data.get_positions(replica_idx=k) for k in range(n_beads)])
spread = np.linalg.norm(beads - beads.mean(axis=0), axis=-1)
spread = spread.mean(axis=(0, 1)) / spk_units.unit2internal("Angstrom")

symbols = np.array(molecule.get_chemical_symbols())
for element in ["H", "C", "O"]:
    print(f"mean bead spread {element}: {spread[symbols == element].mean():.3f} Å")

# %% [markdown]
# Over long runs, RPMD also shifts vibrational bands: the O–H and C–H stretches
# move to lower wavenumbers and closer to experiment. Power spectra of RPMD logs
# are computed exactly as above.
#
# ## MD with ASE
#
# The ASE interface runs SchNetPack models in ASE's MD and optimizers.
# {class}`~schnetpack.interfaces.ase_interface.SpkCalculator` is an ASE calculator
# for one structure at a time. It builds the
# neighbor list for each structure and converts the model's units to ASE's eV and
# Å. Here it drives ASE's Langevin dynamics at 300 K for 500 steps of 0.5 fs.

# %%
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

atoms = molecule.copy()
atoms.calc = spk.interfaces.SpkCalculator(
    model=model_path,
    neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff),
    energy_unit="kcal/mol",
    position_unit="Angstrom",
    device=device,
)

rng = np.random.default_rng(0)
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
dynamics = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=temperature,
    friction=0.01 / units.fs,
    rng=rng,
)

ase_temperatures = []
dynamics.attach(lambda: ase_temperatures.append(atoms.get_temperature()), interval=10)
dynamics.run(500)
print(f"mean temperature: {np.mean(ase_temperatures):.0f} K")

# %% [markdown]
# ASE is the simpler route for a single structure, and it offers ASE's whole set of
# integrators and optimizers. `schnetpack.md` is the one for GPUs, many molecules
# or replicas at once, ring-polymer MD and custom hooks.
#
# ## Summary
#
# A simulation in `schnetpack.md` is a `System`, an integrator and a calculator,
# run by a `Simulator`. Hooks add thermostats, logging and checkpoints, and
# `HDF5Loader` reads the log back for analysis. Ring-polymer MD needs only more
# replicas, another integrator and another thermostat. For routine runs, the
# [`spkmd` command line](../guides/md_cli.md) sets all of this up from config
# files.

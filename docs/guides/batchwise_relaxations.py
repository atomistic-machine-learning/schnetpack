# %% [markdown]
# # Batch-wise Structure Relaxation

# %% [markdown]
# In this tutorial, we show how to use the ``LBFGS`` optimizer of ``schnetpack.dynamics``. It relaxes structures in a batch-wise manner, i.e. it optimizes multiple structures in parallel, keeping one inverse Hessian approximation per structure. This is particularly useful, when many relatively similar structures (--> similar time until convergence) should be relaxed while requiring possibly short simulation time.

# %%
import numpy as np
import torch
from ase.build import minimize_rotation_and_translation
from ase.io import read
from scipy.optimize import linear_sum_assignment

import schnetpack as spk
from schnetpack import properties
from schnetpack.dynamics import ForceCalculator
from schnetpack.interfaces.ase_interface import atoms_to_batch, batch_to_atoms
from schnetpack.utils.compatibility import load_model

# %% [markdown]
# First, we load the force field model that provides the forces for the relaxation process. Furthermore, we define the atoms converter, which is used to convert ase Atoms objects to SchNetPack input.
#
# Please note: The benefit from using batch-wise relaxations highly depends on your hardware setup. It is significantly faster on a cuda device than cpu. A benchmarking test is implemented in `tests/dynamics/relax/test_relaxer_benchmark.py`, see the Benchmarking section at the end. That both optimizers land in the same minima is checked in `tests/dynamics/relax/test_relaxer_vs_ase.py`.

# %%
model_path = "../../tests/testdata/md_ethanol.model"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = load_model(model_path, device=device)

cutoff = model.representation.cutoff.item()
CUTOFF_SKIN = 0.3  # Ang


# settings shared by every relaxation in this notebook
ENERGY_UNIT = "kcal/mol"
POSITION_UNIT = "Ang"
FMAX = 0.001  # converged once no force exceeds this, in eV/Ang
MAX_STEPS = 1000  # give up after this many optimizer steps
NOISE = 0.05  # standard deviation of the Gaussian noise in Angstrom (added to atomic positions)
SEED = 0


def batchwise_calculator(model=model_path):
    """A calculator that evaluates a whole batch of structures in a single model call.

    ``model`` is either a path or a ready model object. Restraints are added to the
    energy surface as guidance on the calculator, see ``howto_guidance.ipynb``.
    The model works in kcal/mol and Angstrom; the calculator converts, so the optimizer
    sees eV and Angstrom. Forces are the derivative of the energy, so a physical
    force calculator runs the model with autograd enabled.
    """
    return ForceCalculator(
        model,
        neighbor_list=spk.dynamics.BatchNeighborList(
            neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff),
            cutoff_skin=CUTOFF_SKIN,
        ),
        device=device,
        dtype=torch.float32,
        energy_unit=ENERGY_UNIT,
        position_unit=POSITION_UNIT,
    )


def batchwise_relaxer(model=model_path, **kwargs):
    """An ``LBFGS`` optimizer that relaxes a whole batch at once, until ``FMAX``.

    Builds its own calculator, so every call starts from a clean state. The calculator
    converts the model's units, so ``FMAX`` is in eV/Ang. Remaining keyword arguments
    are passed on to ``LBFGS`` (``maxstep``, ``hooks``, ...).
    """
    return spk.dynamics.LBFGS(batchwise_calculator(model), fmax=FMAX, **kwargs)


def make_structures(n_structures, seed=None):
    """``n_structures`` starting geometries, cycling through the conformers.

    Every copy gets Gaussian noise on its atomic positions, so the structures of a batch
    are similar but not identical. Pass a ``seed`` to get a reproducible set.
    """
    rng = np.random.default_rng(seed)
    structures = []
    for idx in range(n_structures):
        noisy_atoms = conformers[idx % len(conformers)].copy()
        noisy_atoms.positions += rng.normal(
            scale=NOISE, size=noisy_atoms.positions.shape
        )
        structures.append(noisy_atoms)
    return structures


# %% [markdown]
# Subsequently, we load the batch of initial structures utilizing ASE (supports xyz, db and more).

# %%
input_structure_file = "../../tests/testdata/ethanol_conformers.xyz"

# load the initial structures
conformers = read(input_structure_file, index=":")

n_replica = 5
ats = make_structures(n_replica * len(conformers), seed=SEED)

# %% [markdown]
# The positions of certain atoms during the relaxation can be fixed by flagging them in the batch: ``batch[properties.fixed_atoms]``. It holds one boolean per atom of the batch ($n_\text{atoms} \times n_\text{structures}$ entries), ``True`` for every atom whose position is held fixed. It is the same key the ``FreezeScaffold`` hook reads. Here, we do not fix any atoms, so the mask contains only ``False``.

# %%
# one bool per atom of the batch: True holds the atom in place, False lets it move
n_atoms = len(ats[0].get_atomic_numbers())
single_structure_mask = torch.zeros(n_atoms, dtype=torch.bool)
# the same atoms are fixed in every structure of the batch
mask = single_structure_mask.repeat(len(ats))

# %% [markdown]
# Finally, we run the optimization:

# %%
relaxer = batchwise_relaxer()

# run optimization
batch = atoms_to_batch(ats, device=device)
batch[properties.fixed_atoms] = mask.to(device)
relaxed = relaxer.run(batch, MAX_STEPS)

# the relaxer stops after MAX_STEPS even if the batch has not converged, so check it did
max_force = relaxer.calculator(relaxed)["forces"].norm(dim=-1).max()
assert max_force < FMAX, "the relaxation hit the step limit"

# the relaxer works on tensors; batch_to_atoms converts the result back to ase
bw_atoms = batch_to_atoms(relaxed)

# %% [markdown]
# ## Relaxing with MACE
#
# The batch-wise optimizer is not tied to SchNetPack models: it runs anything that returns energies and forces for a batch. `MaceModel` puts an exported [MACE](https://github.com/ACEsuit/mace) model behind SchNetPack's batch format, so the same `LBFGS` relaxes the same structures on a second potential energy surface. Here that is MACE-OFF23 (medium), a foundation model for organic molecules, and we compare its relaxed structures with the SchNetPack ones from above.
#
# `export_mace_off` downloads MACE-OFF and exports it with `torch.export`, which needs `mace-torch`: `pip install "schnetpack[mace]"`. The exported model is cached, and loading it needs no `mace-torch`. Mind the license of the weights: MACE-OFF is licensed for academic, non-commercial use only (ASL).
#
# MACE-OFF is a much larger model than the PaiNN model above, and each step costs correspondingly more.

# %%
from schnetpack.interfaces.mace_interface import export_mace_off
from schnetpack.model import MaceModel, load_mace

# MACE works in eV and Angstrom, the calculator's defaults. The model is cast here and
# not by the calculator: MaceModel casts only MACE itself, so its per-element reference
# energies stay in float64
mace_model = MaceModel(load_mace(export_mace_off("medium")), dtype="float32")


def mace_calculator():
    """Like ``batchwise_calculator``, on MACE-OFF: the neighbor list covers MACE's cutoff."""
    return ForceCalculator(
        mace_model,
        neighbor_list=spk.dynamics.BatchNeighborList(
            neighbor_list=spk.transform.MatScipyNeighborList(cutoff=mace_model.r_max),
            cutoff_skin=CUTOFF_SKIN,
        ),
        device=device,
    )


# %% [markdown]
# We relax the very same batch of noisy structures as above, with the same optimizer settings:

# %%
mace_relaxer = spk.dynamics.LBFGS(mace_calculator(), fmax=FMAX)

batch = atoms_to_batch(ats, device=device)
batch[properties.fixed_atoms] = mask.to(device)
mace_relaxed = mace_relaxer.run(batch, MAX_STEPS)

max_force = mace_relaxer.calculator(mace_relaxed)["forces"].norm(dim=-1).max()
assert max_force < FMAX, "the MACE relaxation hit the step limit"
mace_atoms = batch_to_atoms(mace_relaxed)


# %% [markdown]
# ## Comparing MACE and SchNetPack
#
# Every conformer of the input file was relaxed in several noisy replicas. Per conformer we check:
#
# * **Replicas**: all replicas of a conformer should relax into the same minimum. The spread of their energies, taken on each model separately, says whether they did.
# * **Geometry**: the RMSD between the MACE and the SchNetPack minimum.
#
# Two effects make a plain RMSD misleading. A relaxation may translate and rotate the molecule as a whole -- removed with ASE's `minimize_rotation_and_translation`. And identical atoms are indistinguishable: ethanol's methyl group rotates freely, so a 120° methyl rotation returns the same physical structure with the three hydrogens relabelled. Pairing atoms by index would report a large deviation for two structures that are in fact identical.
#
# We therefore use a **permutation-matched RMSD**, minimised over rigid-body motion *and* relabellings of like atoms (Hungarian assignment via `scipy.optimize.linear_sum_assignment`, alternated with the superposition until the assignment stops changing).


# %%
def rmsd(P, Q):
    return np.sqrt(((P - Q) ** 2).sum() / len(P))


def matched_rmsd(atoms, reference, max_iter=20):
    """RMSD minimized over rigid-body motion *and* relabellings of atoms with identical type.

    Alternates the Kabsch superposition with an optimal assignment of identical
    elements until the permutation stops changing. Returns (rmsd, permutation),
    where permutation[i] is the atom of `atoms` matched to atom i of `reference`.
    """
    ref_pos = reference.get_positions()
    symbols = np.array(reference.get_chemical_symbols())
    groups = [np.where(symbols == s)[0] for s in sorted(set(symbols))]

    perm = np.arange(len(atoms))
    for _ in range(max_iter):
        current = atoms.copy()
        current.positions = atoms.get_positions()[perm]
        minimize_rotation_and_translation(reference, current)
        pos = current.get_positions()

        new_perm = perm.copy()
        for idx in groups:
            cost = ((pos[idx][:, None, :] - ref_pos[None, idx, :]) ** 2).sum(-1)
            row, col = linear_sum_assignment(cost)
            new_perm[idx[col]] = perm[idx[row]]

        if np.array_equal(new_perm, perm):
            return rmsd(pos, ref_pos), perm
        perm = new_perm

    raise RuntimeError("permutation did not converge")


# %%
def evaluate_energies(calculator, atoms_list):
    """Energies [eV] per structure, in one call."""
    batch = atoms_to_batch(atoms_list, device=device)
    return calculator.energy(batch).detach().cpu().numpy()


e_spk = evaluate_energies(batchwise_calculator(), bw_atoms)
e_mace = evaluate_energies(mace_calculator(), mace_atoms)
rmsd_models = np.array([matched_rmsd(m, s)[0] for m, s in zip(mace_atoms, bw_atoms)])
conformer_of = np.arange(len(ats)) % len(conformers)

header = (
    f"{'conformer':>9} {'RMSD MACE vs SchNetPack':>24} "
    f"{'replica spread SchNetPack':>26} {'replica spread MACE':>20}"
)
print(header)
print(f"{'':>9} {'[A]':>24} {'[meV]':>26} {'[meV]':>20}")
print("-" * len(header))
for c in range(len(conformers)):
    sel = conformer_of == c
    print(
        f"{c:>9} {rmsd_models[sel].mean():>24.2e} "
        f"{np.ptp(e_spk[sel]) * 1e3:>26.2f} {np.ptp(e_mace[sel]) * 1e3:>20.2f}"
    )

# %% [markdown]
# * **The replicas agree.** All copies of a conformer relax into the same minimum on either model, to within hundredths of a meV.
# * **The geometries agree closely.** The MACE and SchNetPack minima of each conformer lie only one to two hundredths of an Angstrom apart.

# %% [markdown]
# ## Benchmarking
#
# How much time the batch-wise optimizer saves over relaxing one structure at a time depends on the device, the model and the size of the structures. `tests/dynamics/relax/test_relaxer_benchmark.py` times the batch-wise optimizer against sequential ase `LBFGS` over a range of batch sizes. The benchmark is deselected by default; run it from the repository root with
#
# ```bash
# pytest tests/dynamics/relax -m benchmark_sweep --benchmark-group-by=param --benchmark-time-unit=ms
# ```

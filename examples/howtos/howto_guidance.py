# %% [markdown]
# # Restraining a Relaxation

# %% [markdown]
# A restraint adds a term to the energy surface a relaxation follows, without touching the model. In `schnetpack.dynamics` it is *guidance* on the calculator: the `ForceCalculator` evaluates it at every point it evaluates the model and adds its forces to the model's forces, so the optimizer only ever sees the restrained forces. The restraint drives the steps and counts towards `fmax` like any other force.
#
# `HarmonicRestraint` restrains the distances of atom pairs,
#
# $$E = \tfrac{1}{2}\, k\, (d - d_0)^2,$$
#
# summed per structure. Pairs, target distances and force constants are not arguments of the module but keys of the batch, so they collate with the structures, and every structure of a batch may carry its own restraints, or none. Its forward pass takes a batch and returns the restraint's `forces` per atom, in eV/Å; its `energy` method gives the energy per structure. The batch itself is left unchanged. Guidance returns no energy in general, only a force-like term, so the same contract also carries guidance that is no energy (kT ∇ log p) when sampling, on a `GenerativeCalculator`.

# %%
import torch
from ase.io import read

import schnetpack as spk
from schnetpack import properties
from schnetpack.dynamics import LBFGS, ForceCalculator, HarmonicRestraint
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.utils.compatibility import load_model

# %% [markdown]
# We load the force field model and build a relaxer around it. The calculator evaluates the whole batch in one model call; the `ForceCalculator` converts the model's kcal/mol to eV, so the optimizer, `FMAX` and the restraint all work in eV and Angstrom.

# %%
model_path = "../../tests/testdata/md_ethanol.model"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = load_model(model_path, device=device)
cutoff = model.representation.cutoff.item()
CUTOFF_SKIN = 0.3  # Ang

ENERGY_UNIT = "kcal/mol"
POSITION_UNIT = "Ang"
FMAX = 0.001  # converged once no force exceeds this, in eV/Ang
MAX_STEPS = 1000  # give up after this many optimizer steps


def relaxer(guidance=()):
    """An ``LBFGS`` optimizer on a fresh calculator carrying ``guidance``."""
    calculator = ForceCalculator(
        model_path,
        neighbor_list=spk.dynamics.BatchNeighborList(
            neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff),
            cutoff_skin=CUTOFF_SKIN,
        ),
        device=device,
        dtype=torch.float32,
        energy_unit=ENERGY_UNIT,
        position_unit=POSITION_UNIT,
        guidance=guidance,
    )
    return LBFGS(calculator, fmax=FMAX)


# %% [markdown]
# The demo below relaxes ethanol's C-O bond starting from a few different conformers.

# %%
input_structure_file = "../../tests/testdata/ethanol_conformers.xyz"

# load the starting structures
conformers = read(input_structure_file, index=":")
batch = atoms_to_batch(conformers, device=device)

# %% [markdown]
# Ethanol comes out of the file as `C C O H H H H H H`, so atoms `(0, 2)` are the C-O bond. It relaxes to about 1.43 Å on its own; we restrain it to 1.8 Å. Every structure carries one restraint, so the batch gets
#
# - `HarmonicRestraint.n_restraints` — the number of restrained pairs per structure,
# - `HarmonicRestraint.restraint_pairs` — the pairs of all structures end to end, with atom indices local to their structure,
# - `HarmonicRestraint.restraint_lengths` and `HarmonicRestraint.restraint_constants` — one target distance (Å) and force constant (eV/Å²) per pair.

# %%
BOND = (0, 2)  # the C-O bond of ethanol
BOND_LENGTH = 1.8  # Angstrom, well beyond the equilibrium C-O distance
FORCE_CONSTANT = 100.0  # eV / Angstrom**2

n_structures = len(conformers)
restrained_batch = {
    **batch,
    HarmonicRestraint.n_restraints: torch.ones(n_structures, dtype=torch.long),
    HarmonicRestraint.restraint_pairs: torch.tensor([BOND] * n_structures),
    HarmonicRestraint.restraint_lengths: torch.full((n_structures,), BOND_LENGTH),
    HarmonicRestraint.restraint_constants: torch.full((n_structures,), FORCE_CONSTANT),
}


# %% [markdown]
# We relax the same structures twice: once without guidance, and once with a `HarmonicRestraint` as the calculator's guidance.


# %%
def bond_lengths(batch):
    """The C-O bond length of every structure of ``batch``."""
    positions = batch[properties.R].reshape(n_structures, -1, 3)
    return (
        torch.norm(positions[:, BOND[1]] - positions[:, BOND[0]], dim=-1).cpu().numpy()
    )


free = relaxer().run(batch, MAX_STEPS)
restrained = relaxer(guidance=[HarmonicRestraint()]).run(restrained_batch, MAX_STEPS)

print(f"{'':<14}{'C-O bond [Ang]':>16}")
for label, relaxed in [("unrestrained", free), ("restrained", restrained)]:
    print(f"{label:<14}{bond_lengths(relaxed).mean():>16.3f}")
print(f"{'target':<14}{BOND_LENGTH:>16.3f}")

# %% [markdown]
# The bond settles where the restraint's pull balances the model's: the stiffer the force constant, the closer to the target. The restraint's `energy` method gives its own energy per structure, e.g. to compare them with the model's energy of the relaxed structures:

# %%
energy = HarmonicRestraint().energy(restrained)
print("restraint energy [eV]:", energy.cpu().numpy())
print(
    "model energy [eV]:    ",
    relaxer().calculator(restrained)[properties.energy].cpu().numpy(),
)

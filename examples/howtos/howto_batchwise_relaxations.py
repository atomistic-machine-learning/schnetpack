# %% [markdown]
# ## <span style="color:red">Batchwise structure optimization is deprecated and will be upgraded soon</span>
#

# %% [markdown]
# # Batch-wise Structure Relaxation

# %% [markdown]
# In this tutorial, we show how to use the ``ASEBatchwiseLBFGS``. It enables relaxation of structures in a batch-wise manner, i.e. it optimizes multiple structures in parallel. This is particularly useful, when many relatively similar structures (--> similar time until convergence) should be relaxed while requiring possibly short simulation time.

# %%
# import os
# import shutil

# import torch
# from ase.io import read

# import schnetpack as spk
# from schnetpack import properties
# from schnetpack.interfaces.ase_interface import AtomsConverter
# from schnetpack.interfaces.batchwise_optimization import ASEBatchwiseLBFGS, BatchwiseCalculator

# %% [markdown]
# First, we load the force field model that provides the forces for the relaxation process. Furthermore, we define the atoms converter, which is used to convert ase Atoms objects to SchNetPack input. Eventually the calculator is initialized. The latter provides the necessary functionality to load a model and calculates forces and energy for the respective structures. Please note that running batchwise relaxations is significantly faster on a cuda device.

# %%
# model_path = "../../tests/testdata/md_ethanol.model"

## set device
# device = torch.device("cpu")

## load model
# model = torch.load(model_path, map_location=device, weights_only=False)

## define neighbor list
# cutoff = model.representation.cutoff.item()
# nbh_list=spk.transform.MatScipyNeighborList(cutoff=cutoff)

## build atoms converter
# atoms_converter = AtomsConverter(
#    neighbor_list=nbh_list,
#    device=device,
# )

## build calculator
# calculator = BatchwiseCalculator(
#    model=model_path,
#    atoms_converter=atoms_converter,
#    device=device,
#    energy_unit="kcal/mol",
#    position_unit="Ang",
# )

# %% [markdown]
# Subsequently, we load the batch of initial structures utilizing ASE (supports xyz, db and more).

# %%
# input_structure_file = "../../tests/testdata/md_ethanol.xyz"

## load initial structures
# ats = read(input_structure_file, index=":")

# %% [markdown]
# For some systems it helps to fix the positions of certain atoms during the relaxation. This can be achieved by providing a mask of boolean entries to ``ASEBatchwiseLBFGS``. The mask is a list of $n_\text{atoms}$ entries, indicating atoms, which positions are fixed during the relaxation. Here, we do not fix any atoms. Hence, the mask only contains ``True``.

# %%
## define structure mask for optimization (True for fixed, False for non-fixed)
# n_atoms = len(ats[0].get_atomic_numbers())
# single_structure_mask = [False for _ in range(n_atoms)]
## expand mask by number of input structures (fixed atoms are equivalent for all input structures)
# mask = single_structure_mask * len(ats)

# %% [markdown]
# Finally, we run the optimization:

# %%
# results_dir = "./howto_batchwise_relaxations_outputs"
# if not os.path.exists(results_dir):
#    os.makedirs(results_dir)

## Initialize optimizer
# optimizer = ASEBatchwiseLBFGS(
#    calculator=calculator,
#    atoms=ats,
#    trajectory="./howto_batchwise_relaxations_outputs/relax_traj",
# )

## run optimization
# optimizer.run(fmax=0.0005, steps=1000)

# %%
# if os.path.exists(results_dir):
#    shutil.rmtree(results_dir)

# %% [markdown]
# Optimzed structures (in the form of ASE `Atoms`) and properties can be obtained with the `get_relaxation_results` function.

# %%
## get list of optimized structures and properties
# opt_atoms, opt_props = optimizer.get_relaxation_results()

# for oatoms in opt_atoms:
#    print(oatoms.get_positions())

# print(opt_props)

# %%

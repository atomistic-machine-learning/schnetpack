
#Batch-wise Structure Relaxation

#In this tutorial, we show how to use the ASEBatchwiseLBFGS. It enables relaxation of structures in a batch-wise manner, i.e. it optimizes multiple structures in parallel. This is particularly useful, when many relatively similar structures (--> similar time until convergence) should be relaxed while requiring possibly short simulation time.

import os
import torch
import random
import shutil
import time
import numpy as np
from tqdm import tqdm 
from matplotlib import pyplot as plt

import ase
from ase.io import read, write
from ase import Atoms
from ase.optimize.lbfgs import LBFGS

import schnetpack as spk
from schnetpack import properties
from schnetpack.interfaces.ase_interface import AtomsConverter, AseInterface
from schnetpack.interfaces.batchwise_optimization import ASEBatchwiseLBFGS, BatchwiseCalculator

#First, we load the force field model that provides the forces for the relaxation process. Furthermore, we define the atoms converter, which is used to convert ase Atoms objects to SchNetPack input. Eventually the calculator is initialized. The latter provides the necessary functionality to load a model and calculates forces and energy for the respective structures.

random.seed(42)
batch_size = 30
batch_mode = "random_distortion"

model_path = "tests/testdata/md_ethanol2.model"

# set device
device = torch.device("cuda")

# load model
model = torch.load(model_path, map_location=device)

# define neighbor list
cutoff = model.representation.cutoff.item()
nbh_list=spk.transform.MatScipyNeighborList(cutoff=cutoff)

# build atoms converter
atoms_converter = AtomsConverter(
    neighbor_list=nbh_list,
    device=device,
)

# build calculator
calculator = BatchwiseCalculator(
    model=model_path,
    atoms_converter=atoms_converter,
    device=device,
    energy_unit="kcal/mol",
    position_unit="Ang",
    dtype=torch.float32
)

#Subsequently, we load an initial ethanol structure utilizing ASE (supports xyz, db and more) and create a batch of initial structures by random distortion.

if not os.path.exists('howto_batchwise_relaxations_outputs'):
    os.makedirs('howto_batchwise_relaxations_outputs')
    
if batch_mode == "md_samples":
    
    ats = read("../../tests/testdata/md_ethanol_batch.xyz", index=":")
    for at_idx, at in enumerate(ats):
        write("./howto_batchwise_relaxations_outputs/init_ethanol_{}.xyz".format(at_idx), at, format="xyz")  
        
elif batch_mode == "random_distortion":
    
    input_structure_file = "tests/testdata/md_ethanol.xyz"

    # load initial structure
    mol = read(input_structure_file)
    pos = mol.get_positions()
    # distort the structures and store them
    for at_idx in range(batch_size):
        for n in range(pos.shape[0]):
            pos[n] = pos[n] * random.uniform(0.95,1.15)
        at = Atoms(positions=pos, numbers=mol.get_atomic_numbers())
        write("./howto_batchwise_relaxations_outputs/init_ethanol_{}.xyz".format(at_idx), at, format="xyz")    
        pos = mol.get_positions()

    # get list of initial structures
    ats = []
    for at_idx in range(batch_size):
        ats.append(read("./howto_batchwise_relaxations_outputs/init_ethanol_{}.xyz".format(at_idx)))

#For some systems it helps to fix the positions of certain atoms during the relaxation. This can be achieved by providing a mask of boolean entries to ASEBatchwiseLBFGS. The mask is a list of

#entries, indicating atoms, which positions are fixed during the relaxation. Here, we do not fix any atoms. Hence, the mask only contains False.

# define structure mask for optimization (True for fixed, False for non-fixed)
n_atoms = len(ats[0].get_atomic_numbers())
single_structure_mask = [False for _ in range(n_atoms)]
# expand mask by number of input structures (fixed atoms are equivalent for all input structures)
mask = single_structure_mask * len(ats)

#Finally, we run the optimization:

# Initialize optimizer
optimizer = ASEBatchwiseLBFGS(
    calculator=calculator,
    atoms=ats,
    trajectory="./howto_batchwise_relaxations_outputs/relax_traj",
    log_every_step=False,
    fixed_atoms_mask=mask
)

# run optimization
t_start_bw = time.time()
optimizer.run(fmax=0.0005, steps=1000)
t_end_bw = time.time()
relaxation_time_bw = t_end_bw - t_start_bw

#To show the advantage of batch-wise relaxations, in the following, we run the structure optimizations sequentially and compare the total computation time both approaches (batch-wise and sequentially).

# run individual structure optimizations as reference
t_start_indiv = time.time()
for at_idx in range(batch_size):

    relax_dir = "howto_batchwise_relaxations_outputs/relax_{}".format(at_idx)
    if os.path.exists(relax_dir):
        shutil.rmtree(relax_dir)
    os.makedirs(relax_dir)

    ase_interface = AseInterface(
        molecule_path="./howto_batchwise_relaxations_outputs/init_ethanol_{}.xyz".format(at_idx),
        working_dir=relax_dir,
        model_file=model_path,
        neighbor_list=nbh_list,
        device=device,
        dtype=torch.float32,
        energy_unit="kcal/mol",
        position_unit="Ang",
        optimizer_class=LBFGS,
    )
    ase_interface.optimize(fmax=0.0005, steps=1000)

t_end_indiv = time.time()
relaxation_time_indiv = t_end_indiv - t_start_indiv

# print out time difference
print("\n" \
      "The batch-wise relaxation took %.3f seconds on your %s device\n" \
      "while individual structure relaxations would take %.3f seconds on the same device.\n" \
      "You saved %.3f seconds" % 
      (relaxation_time_bw, device, relaxation_time_indiv, relaxation_time_indiv - relaxation_time_bw))

#Optimzed structures (in the form of ASE Atoms) and properties can be obtained with the get_relaxation_results function.

# get list of optimized structures and properties
opt_atoms, opt_props = optimizer.get_relaxation_results()

for oatoms in opt_atoms:
    print(oatoms.get_positions())
    
print(opt_props)

#Neglecting small variations due to numerical errors, both approaches provide the same results. We verify this by comparing the root mean square distances (RMSD) between the relaxed structures.

# get all rmsds
rmsds = []
for traj_idx in range(batch_size):

    traj_file_1 = "./howto_batchwise_relaxations_outputs/relax_traj_{}.xyz".format(traj_idx)
    traj_file_2 = "./howto_batchwise_relaxations_outputs/relax_{}/optimization.traj".format(traj_idx)

    traj_ats_1 = read(traj_file_1, index="-1:")
    traj_ats_2 = read(traj_file_2, index="-1:")

    for step_idx, (at_i, at_j) in enumerate(zip(traj_ats_1, traj_ats_2)):
        # rotate and translate to minimize rmsd
        ase.build.minimize_rotation_and_translation(at_i, at_j)

        # store reference and current structure in one atom object for visual comparison with "ase gui"
        pos_store = np.concatenate((at_i.get_positions(), at_j.get_positions()), axis=0)
        at_nums_store = np.concatenate((at_i.get_atomic_numbers(), at_j.get_atomic_numbers()), axis=0)
        at_store = Atoms(positions=pos_store, numbers=at_nums_store)
        write(
            os.path.join("./howto_batchwise_relaxations_outputs/comp.xyz"),
            at_store,
            format="extxyz",
            append=False if step_idx == 0 and traj_idx == 0 else True,
        )

        # calculate rmsd
        squared_distances = (at_i.get_positions() - at_j.get_positions()) ** 2
        rmsds.append(squared_distances.sum(1).mean())

print("RMSD_max = %.2fe-9 A" % (max(rmsds) * 1e9))

figure = plt.figure()
plt.scatter(range(len(rmsds)), rmsds)
plt.xlabel("sample index")
plt.ylabel("RMSD (Angstrom)")
plt.show()

#Now we concatenate the relaxation results of the batch-wise and sequential relaxation, respectively.

ats_sequ = []
ats_batch = []
for traj_idx in range(batch_size):
    
    traj_file_1 = "./howto_batchwise_relaxations_outputs/relax_traj_{}.xyz".format(traj_idx)
    traj_file_2 = "./howto_batchwise_relaxations_outputs/relax_{}/optimization.traj".format(traj_idx)
    
    ats_batch.append(read(traj_file_1))
    ats_sequ.append(read(traj_file_2))
    
write(
    os.path.join("./howto_batchwise_relaxations_outputs/sequ_ats.xyz"),
    ats_sequ,
    format="extxyz",
)
write(
    os.path.join("./howto_batchwise_relaxations_outputs/batch_ats.xyz"),
    ats_batch,
    format="extxyz",
)

 


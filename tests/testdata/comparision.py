# %%
import numpy as np
import ase
from ase.io import read, write
from ase import Atoms
import os
import matplotlib.pyplot as plt
# %%
rmsds = []

path = "/home/elron/phd/benchmark_data/ethanol"

tags = [ "ase-cpu-opt.xyz",
"batchwise-cpu-opt.xyz",
"ensemble-cpu-opt.xyz",
"ase-cuda-opt.xyz",
"batchwise-cuda-opt.xyz",
 "ensemble-cuda-opt.xyz",
]

rmsd_d = {}

for t1 in tags:
    for t2 in tags:

        traj_ats_1 = read(os.path.join(path,t1),index = ":")
        traj_ats_2 = read(os.path.join(path,t2), index = ":")

        key_1 = t1.split("-opt.xyz")[0]
        key_2 = t2.split("-opt.xyz")[0]

        key = key_1 + " vs " + key_2

        for step_idx, (at_i, at_j) in enumerate(zip(traj_ats_1, traj_ats_2)):


            pos_store = np.concatenate((at_i.get_positions(), at_j.get_positions()), axis=0)
            at_nums_store = np.concatenate((at_i.get_atomic_numbers(), at_j.get_atomic_numbers()), axis=0)
            at_store = Atoms(positions=pos_store, numbers=at_nums_store)

            # calculate rmsd
            squared_distances = (at_i.get_positions() - at_j.get_positions()) ** 2
            rmsds.append(squared_distances.sum(1).mean())
        rmsd_d[key] = rmsds
        rmsds = []





# %%

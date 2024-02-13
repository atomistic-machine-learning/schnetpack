'''
# =================================================================
# INTENTION
# =================================================================

Visualziation of the performance for the
- Ag3+/Ag3- SchNet representation trained model with electronic embedding
- singlet-CH2 / triplet-CH2 SchNet representation trained model with electronic embedding

Evaluation will be done by plotting the PES 
of the four molecules. If they are qualitative similar, to the SpookyNet
results we will implement the embedding for the PaiNN model.


[x] Because error in script no best inference model is saved just the best checkpoint
    - the last checkpoint is saved as best model called checkpoints
    - checked by comparing the weights of the model modules with best_checkpoint. The weights are all equal


# =================================================================
# USAGE
# =================================================================

'''


# =================================================================
# IMPORTS
# =================================================================

import os
import numpy as np
import torch
from schnetpack.data.atoms import ASEAtomsData
from schnetpack.data import load_dataset,AtomsDataFormat,AtomsLoader
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.task import AtomisticTask
from ase import Atoms
from tqdm import tqdm
import pickle
# =================================================================
# Condtions
# =================================================================

#dict to store the results
results = {
    "Ag3+":{"reference_calc":{"energy":None},"model_calc":{"energy":None}},
    "Ag3-":{"reference_calc":{"energy":None},"model_calc":{"energy":None}},
    "singlet-CH2":{"reference_calc":{"energy":None},"model_calc":{"energy":None}},
    "triplet-CH2":{"reference_calc":{"energy":None},"model_calc":{"energy":None}}
}

# conditions for the PES (e.g we plot alpha vs. r for every specific E(alpha,r)) and all needed files
conditions = {
    "Ag3+": {
        "r": [2.6,3.0],
        "alpha": [45,75],
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/ase_ag3_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/checkpoints",
        "check_condition":1.0},
    "Ag3-": {
        "r": [2.5,2.8],
        "alpha": [165,195],
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/ase_ag3_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/checkpoints",
        "check_condition":-1.0},
    "singlet-CH2": {
        "r": [1.0,1.2],
        "alpha": [90,140],
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/ase_carbene_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/checkpoints",
        "check_condition":0.0},
    "triplet-CH2": {
        "r": [1.0,1.2],
        "alpha": [90,140],
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/ase_carbene_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/checkpoints",
        "check_condition":2.0}
}

results = {
    "Ag3+":{},
    "Ag3-":{},
    "singlet-CH2":{},
    "triplet-CH2":{}
}

for key in conditions.keys():

    #load the model and set to eval
    chk_point = conditions[key]["model_file"]
    model = torch.load(chk_point,map_location=torch.device('cpu'))
    model.eval()

    #load split file to get test_idx 
    split = np.load(conditions[key]["split_file"])
    test_idx = split["test_idx"]

    #check condition for assignment to correct dict
    check_condition = conditions[key]["check_condition"]
    
    #list to store results
    ref_e = []
    model_e = []
    dists = []
    angles = []
    d = {}

    #load the data
    data = ASEAtomsData(datapath = conditions[key]["database"])

    #iterate over data to get positions, angles, energies and model prediction energies
    for n in tqdm(range(len(data))):
        
        sample = data.__getitem__(n)
        R = sample["_positions"]
        Z = sample["_atomic_numbers"]
        q = sample["total_charge"]
        E = sample["energy"]
        s = sample["spin_multiplicity"]

        if (check_condition == s.item()) or (check_condition == q.item()):

            converter = spk.interfaces.AtomsConverter(
                neighbor_list=trn.ASENeighborList(cutoff=5.),
                additional_inputs={"total_charge":q,"spin_multiplicity":s})
            atoms = Atoms(Z, R)

            #distances
            distances = atoms.get_all_distances()
            #angles
            angle = atoms.get_angles([[0,1,2],[1,0,2],[1,2,0]])

            #get the model prediction
            inputs = converter(atoms)
            res = model(inputs)

            #append to list
            ref_e.append(E.detach().numpy())
            model_e.append(res["energy"].detach().numpy())
            dists.append(distances)
            angles.append(angle)

            
    d = {"reference_calculation":ref_e,
            "model_calculation":model_e,
            "distances":dists,
            "angles":angles,
            "test_idx":test_idx}

    results[key] = d
        
print("Done")

save_path = "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/results.pickle"
with open(save_path,"wb") as f:
    pickle.dump(results,f)




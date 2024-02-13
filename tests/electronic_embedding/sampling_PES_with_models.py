'''
# =================================================================
# INTENTION
# =================================================================

Sampling of the Ag3-ion and carbene PES with the trained models
sampling is done on regular grid
the sampling conditions are listed in the conditions dict

the reference data were generated with GFN2-xTB

[] give explicit reference structure for the sampling starting from the geometry optimized ground structure


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
from copy import deepcopy
from itertools import product
from ase.calculators.orca import ORCA
from ase.units import Hartree
from ase.io import write,read
from ase.optimize import BFGS
from math import pi
from ase.visualize import view
from ase.constraints import FixBondLengths, FixBondLength
# =================================================================
# Condtions
# =================================================================


# =================================================================
# Function definitions
# =================================================================

def calc_ref_energy(r,a,calculator,start_mol):
    mol = deepcopy(start_mol)
    a_start = mol.get_angle(0,1,2)
    mol.set_angle(0,1,2,a_start + a)

    r_1 = mol.get_distance(0,1)

    mol.set_distance(0,1,r_1+r)

    r_2 = mol.get_distance(1,2)

    mol.set_distance(1,2,r_2+r)
    mol.set_calculator(calculator)
    #energy is returned in kcal/mol
    energy = mol.get_potential_energy() / Hartree
    return energy


def get_energy(r,a,converter,model,start_mol):

    #set molecule with specific r and a
    #PES sampling is done for distances 0-1 and 1-2 which will have both the same values
    #the angle is between 0-1-2
    mol = deepcopy(start_mol)
    a_start = mol.get_angle(0,1,2)
    mol.set_angle(0,1,2,a_start + a)

    r_1 = mol.get_distance(0,1)

    mol.set_distance(0,1,r_1+r)

    r_2 = mol.get_distance(1,2)

    mol.set_distance(1,2,r_2+r)

    #get the model prediction
    inputs = converter(mol)
    E = model(inputs)["energy"].detach().cpu().numpy().item()
    return E



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
        "ref_mol_opt": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/Ag3+_xtb2.xyz",
        "mol_spk": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/opt-Ag3+-spk.xyz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/ase_ag3_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/checkpoints",
        "check_condition":"total_charge",
        "check_condition_value":1.0,
        "idx":1000},
    "Ag3-": {
        "r": [2.5,2.8],
        "alpha": [165,195],
        "ref_mol_opt": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/Ag3-_xtb2.xyz",
        "mol_spk": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/opt-Ag3--spk.xyz",
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/ase_ag3_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/ag3-ions/checkpoints",
        "check_condition":"total_charge",
        "check_condition_value":-1.0,
        "idx":0},
    "singlet-CH2": {
        "r": [1.0,1.2],
        "alpha": [90,140],
        "ref_mol_opt": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/s-CH2_xtb2.xyz",
        "mol_spk": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/opt-singlet-CH2-spk.xyz",
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/ase_carbene_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/checkpoints",
        "check_condition":"spin_multiplicity",
        "check_condition_value":0.0,
        "idx":1000},
    "triplet-CH2": {
        "r": [1.0,1.2],
        "alpha": [90,140],
        "ref_mol_opt": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/t-CH2_xtb2.xyz",
        "mol_spk": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/opt-triplet-CH2-spk.xyz",
        "split_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/split.npz",
        "database": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/ase_carbene_2200.db",
        "model_file": "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/carbene/checkpoints",
        "check_condition":"spin_multiplicity",
        "check_condition_value":2.0,
        "idx":0}
}

results = {
    "Ag3+":{},
    "Ag3-":{},
    "singlet-CH2":{},
    "triplet-CH2":{}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

for key in conditions.keys():

    #load the model and set to eval
    chk_point = conditions[key]["model_file"]
    model = torch.load(chk_point,map_location=device)
    model.eval()

    #check if either spin or charge is the condition
    if conditions[key]["check_condition"] == "total_charge":
        q_ = conditions[key]["check_condition_value"]
        s_ = 0.0
        q = torch.tensor([conditions[key]["check_condition_value"]],device=device)
        s = torch.tensor([0.0],device=device)
    elif conditions[key]["check_condition"] == "spin_multiplicity":
        s = torch.tensor([conditions[key]["check_condition_value"]],device=device)
        q = torch.tensor([0.0],device=device)
        s_ = conditions[key]["check_condition_value"]
        q_ = 0.0

    #load the data to get a sample molecule    data = ASEAtomsData(datapath = conditions[key]["database"])
    splits = np.load(conditions[key]["split_file"])["test_idx"]
    splits = np.arange(1,2001,1)
    data = ASEAtomsData(datapath = conditions[key]["database"])

    # just to get the mean of the dataset out commented because not necessary for now
    E = []
    for i in splits:
        if i.item() != 0:
            
            

            if (data.conn.get(i.item()+1).data["total_charge"].item() == q_ ) and (data.conn.get(i.item()+1).data["spin_multiplicity"].item() == s_):
                E.append(data.conn.get(i.item()+1).data["energy"])

    E = np.mean(np.concatenate(np.array(E))).item()
    print(f"{key} energy mean {E}")


    #sampling grid
    grid_points = 25
    energies = np.zeros((grid_points,grid_points))
    ref_energies = np.zeros((grid_points,grid_points))
    ri = np.linspace(conditions[key]["r"][0],conditions[key]["r"][1],grid_points)
    ai = np.linspace(conditions[key]["alpha"][0],conditions[key]["alpha"][1],grid_points)

    #define start mol structure
    #start_mol_specs = data.__getitem__(2000)
    #Z, R = start_mol_specs["_atomic_numbers"].detach().numpy(), start_mol_specs["_positions"].detach().numpy()
    #start_mol = Atoms(Z,R,info={"total_charge":q.detach().cpu().numpy().item(),"spin_multiplicity":s.detach().cpu().numpy().item()})
    start_mol = read(conditions[key]["mol_spk"])
    ref_mol = read(conditions[key]["ref_mol_opt"])

    #init converter to get nhb list indices
    converter = spk.interfaces.AtomsConverter(
                neighbor_list=trn.ASENeighborList(cutoff=5.),
                additional_inputs={"total_charge":q,"spin_multiplicity":s},device=device)


    #iterate over the grid and get the energies
    #iteridcs
    iteridx = list(product(range(grid_points),range(grid_points)))
    mols = []
    for idx in tqdm(iteridx):

        r = ri[idx[0]]
        a = ai[idx[1]]

        #traj of displacement structs
        mol = read(conditions[key]["mol_spk"])
        #ref_mol = read(conditions[key]["ref_mol_opt"])
        #mol = deepcopy(start_mol)
        # a_start = mol.get_angle(0,1,2)
        # r_1 = mol.get_distance(0,1)
        # r_2 = mol.get_distance(1,2)


        mol.set_distance(0,1,r,mic=True,fix=0)
        mol.set_distance(0,2,r,mic=True,fix=0)
        mol.set_angle(2,0,1,a)


        #mols.append(mol)
        #get the energy
        E = get_energy(r,a,converter,model,start_mol)
        #E_ref = calc_ref_energy(r,a,calculator,ref_mol)
        #store the results
        energies[idx[0],idx[1]] = E
        #ref_energies[idx[0],idx[1]] = E_ref

# angles = [mol.get_angle(0,1,2) for mol in mols]
# r1 = [mol.get_distance(0,1) for mol in mols]
# r2 = [mol.get_distance(1,2) for mol in mols]

    #store the results
    res = {"energies":energies,"ref_energies":ref_energies,"r":ri,"alpha":ai}
    results[key] = res



    save_path = "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/PES_sampling_"+key+".pickle"
    with open(save_path,"wb") as f:
        pickle.dump(res,f)




    # #init calculator for reference calculation
    # calculator = ORCA(
    #     label="orca",
    #     task = "SP",
    #     charge=q.detach().cpu().numpy().item(),
    #     spin=s.detach().cpu().numpy().item(),
    #     orcasimpleinput="SP XTB2",
    #     orcablocks="%scf maxiter 100 end")

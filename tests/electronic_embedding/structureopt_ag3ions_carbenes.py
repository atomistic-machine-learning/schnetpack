'''
# =================================================================
# INTENTION
# =================================================================

script for performing geo optimization of the ag3-ions and carbenes
with the trained models.

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
from ase.io import write, read
from ase.optimize import BFGS
from ase.visualize import view

# =================================================================
# Function definitions
# =================================================================

def optimize_structure(
        engine: str,
        mol: Atoms,
        chk_point: str,
        save_path:str,
        key: str,
        q: torch.tensor,
        s: torch.tensor
        ):

    traj_path = os.path.join(save_path,key+"_opt_"+engine+".traj")
    #traj_path = "/".join(chk_point.split("/")[:-1])+"/"+str(key)+"-"+engine+".traj"
    #new_mol_path = "/".join(chk_point.split("/")[:-1])+"/opt-"+str(key)+"-"+engine+".xyz"
    #view(mol)
    if engine == "spk":
    #init spk Calculator
        calculator = spk.interfaces.SpkCalculator(
        model_file=chk_point,
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        energy_key="energy",
        force_key="forces",
        energy_unit="eV",
        position_unit="eV/Ang",
        additional_inputs={"total_charge":q,"spin_multiplicity":s})

    if engine == "orca":
    #init ORCA calculator
        calculator = ORCA(
            label="orca",
            task = "OPT",
            charge=q.detach().cpu().numpy().item(),
            spin=s.detach().cpu().numpy().item(),
            orcasimpleinput="XTB2 OPT")
        
    
    #set calculator and run geometry optimization
    mol.set_calculator(calculator)
    dyn = BFGS(mol,trajectory=traj_path)
    dyn.run(fmax=0.0001)
    #write(new_mol_path,mol)

# conditions for the PES (e.g we plot alpha vs. r for every specific E(alpha,r)) and all needed files
conditions = {}


conditions["Ag3-_embed"] = {
    "r": [2.5, 2.8],
    "alpha": [165, 195],
    "base": "/xtb_tmp_calcs/Ag3-",
    "ref_mol_opt": "ag3-ion/optimized_structure/ag3minus_opt_xtb.xyz",
    "mol_spk": "ag3-ion/optimized_structure/ag3minus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "ag3-ions/models/yes_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": -1.0,
    "initial_Geo":"ag3-ions/data/ag3minus_initial_for_GeoOpt.xyz",
}

conditions["Ag3-"] = {
    "r": [2.5, 2.8],
    "alpha": [165, 195],
    "base": "xtb_tmp_calcs/Ag3-",
    "ref_mol_opt": "ag3-ions/optimized_structure/ag3minus_opt_xtb.xyz",
    "mol_spk": "ag3-ions/optimized_structure/ag3minus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "ag3-ions/models/no_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": -1.0,
    "initial_Geo":"ag3-ions/data/ag3minus_initial_for_GeoOpt.xyz",
}

conditions["triplet-CH2_embed"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/triplet-CH2_opt_xtb.xyz",
    "mol_spk": "/carbene/optimized_structure/triplet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/t-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "carbene/models/yes_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 2.0,
    "initial_Geo":"carbene/data/triplet-CH2_initial_for_GeoOpt.xyz",
}


conditions["singlet-CH2_embed"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/singlet-CH2_opt_xtb.xyz",
    "mol_spk": "/carbene/optimized_structure/singlet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/s-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "carbene/models/yes_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 0.0,
    "initial_Geo":"carbene/data/singlet-CH2_initial_for_GeoOpt.xyz",
}

conditions["singlet-CH2"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/singlet-CH2_opt_xtb.xyz",
    "mol_spk": "/carbene/optimized_structure/singlet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/s-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "carbene/models/no_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 0.0,
    "initial_Geo":"carbene/data/singlet-CH2_initial_for_GeoOpt.xyz",
}

conditions["Ag3+_embed"] = {
    "r": [2.6, 3.0],
    "alpha": [45, 75],
    "base": "/xtb_tmp_calcs/Ag3+",
    "ref_mol_opt": "ag3-ion/optimized_structure/ag3plus_opt_xtb.xyz",
    "mol_spk": "ag3-ion/optimized_structure/ag3plus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "ag3-ions/models/yes_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": 1.0,
    "initial_Geo":"ag3-ions/data/ag3plus_initial_for_GeoOpt.xyz",
}

conditions["Ag3+"] = {
    "r": [2.6, 3.0],
    "alpha": [45, 75],
    "base": "xtb_tmp_calcs/Ag3+",
    "ref_mol_opt": "ag3-ions/optimized_structure/ag3plus_opt_xtb.xyz",
    "mol_spk": "ag3-ions/optimized_structure/ag3plus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "ag3-ions/models/no_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": 1.0,
    "initial_Geo":"ag3-ions/data/ag3plus_initial_for_GeoOpt.xyz",
}



conditions["triplet-CH2"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/triplet-CH2_opt_xtb.xyz",
    "mol_spk": "/carbene/optimized_structure/triplet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/t-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "carbene/models/no_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 2.0,
    "initial_Geo":"carbene/data/triplet-CH2_initial_for_GeoOpt.xyz",
}

BASEPATH = "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

for key in conditions.keys():

    #check if either spin or charge is the condition
    if conditions[key]["check_condition"] == "total_charge":
        q = torch.tensor([conditions[key]["check_condition_value"]],device=device)
        s = torch.tensor([0.0],device=device)
    elif conditions[key]["check_condition"] == "spin_multiplicity":
        s = torch.tensor([conditions[key]["check_condition_value"]],device=device)
        q = torch.tensor([0.0],device=device)


    chk_point = os.path.join(BASEPATH,conditions[key]["model_file"])
    model = torch.load(chk_point,map_location=device)
    #model.postprocessors.extend([trn.AddOffsets("energy",add_mean=True,add_atomrefs=False)])
    model.eval()

    
    engine = "spk"
    new_mol_path = os.path.join(BASEPATH,conditions[key]["initial_Geo"])
    opt_mol = read(new_mol_path)

    #init converter to get nhb list indices
    converter = spk.interfaces.AtomsConverter(
                neighbor_list=trn.ASENeighborList(cutoff=5.),
                additional_inputs={"total_charge":q,"spin_multiplicity":s},device=device)

    inputs = converter(opt_mol)
    res = model(inputs)
    #structure optimization with spk model
    mol = read(os.path.join(BASEPATH,conditions[key]["database"]))
    save_path = "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/optimized_structures"
    optimize_structure("spk",opt_mol,chk_point,save_path,key,q,s)
    print("spk optimization done for ",key)
    e_pot = opt_mol.get_potential_energy()
    print("spk energy: ",e_pot)
    # 






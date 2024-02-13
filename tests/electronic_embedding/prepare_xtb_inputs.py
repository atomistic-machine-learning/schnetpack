'''


'''


import os
import numpy as np
from ase.io import read,write
from itertools import product
from tqdm import tqdm
import schnetpack as spk
import schnetpack.transform as trn
import torch
import pickle
from schnetpack.data import ASEAtomsData
from ase.visualize import view  
# conditions for the PES (e.g we plot alpha vs. r for every specific E(alpha,r)) and all needed files
conditions = {}




conditions["singlet-CH2_embed"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/singlet-CH2_opt_xtb.xyz",
    "mol_spk": "carbene/optimized_structure/singlet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/s-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "/carbene/models/yes_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 0.0
}

conditions["singlet-CH2"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/singlet-CH2_opt_xtb.xyz",
    "mol_spk": "carbene/optimized_structure/singlet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/s-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "/carbene/models/no_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 0.0
}

conditions["Ag3+_embed"] = {
    "r": [2.6, 3.0],
    "alpha": [45, 75],
    "base": "xtb_tmp_calcs/Ag3+",
    "ref_mol_opt": "ag3-ions/optimized_structure/ag3plus_opt_xtb.xyz",
    "mol_spk": "ag3-ions/optimized_structure/ag3plus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "/ag3-ions/models/yes_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": 1.0
}

conditions["Ag3+"] = {
    "r": [2.6, 3.0],
    "alpha": [45, 75],
    "base": "xtb_tmp_calcs/Ag3+",
    "ref_mol_opt": "ag3-ions/optimized_structure/ag3plus_opt_xtb.xyz",
    "mol_spk": "ag3-ions/optimized_structure/ag3plus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "/ag3-ions/models/no_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": 1.0
}

conditions["Ag3-_embed"] = {
    "r": [2.5, 2.8],
    "alpha": [165, 195],
    "base": "xtb_tmp_calcs/Ag3-",
    "ref_mol_opt": "ag3-ions/optimized_structure/ag3minus_opt_xtb.xyz",
    "mol_spk": "ag3-ions/optimized_structure/ag3minus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "/ag3-ions/models/yes_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": -1.0
}

conditions["Ag3-"] = {
    "r": [2.5, 2.8],
    "alpha": [165, 195],
    "base": "xtb_tmp_calcs/Ag3-",
    "ref_mol_opt": "ag3-ions/optimized_structure/ag3minus_opt_xtb.xyz",
    "mol_spk": "ag3-ions/optimized_structure/ag3minus_opt_spk.xyz",
    "database": "ag3-ions/data/ase_ag3_2200.db",
    "model_file": "/ag3-ions/models/no_embed/checkpoints",
    "check_condition": "total_charge",
    "check_condition_value": -1.0
}

conditions["triplet-CH2"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/triplet-CH2_opt_xtb.xyz",
    "mol_spk": "carbene/optimized_structure/triplet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/t-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "/carbene/models/no_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 3.0
}

conditions["triplet-CH2_embed"] = {
    "r": [1.0, 1.2],
    "alpha": [90, 140],
    "ref_mol_opt": "/carbene/optimized_structure/triplet-CH2_opt_xtb.xyz",
    "mol_spk": "carbene/optimized_structure/triplet-CH2_opt_spk.xyz",
    "base": "xtb_tmp_calcs/t-CH2",
    "database": "carbene/data/ase_carbene_2200.db",
    "model_file": "/carbene/models/yes_embed/checkpoints",
    "check_condition": "spin_multiplicity",
    "check_condition_value": 3.0
}

 
BASEPATH = "/home/elron/phd/projects/google/experiments/painn_debug_test_q_s_embedding"

def run_orca(folder,out_path):
    command = f"orca {folder} > {out_path}"
    os.system(command)

def get_conditons(conditions,key):

    #check if either spin or charge is the condition
    if conditions[key]["check_condition"] == "total_charge":
        q = conditions[key]["check_condition_value"]
        s = 0.0
        q_ = int(q)
        s_ = int(s)
    elif conditions[key]["check_condition"] == "spin_multiplicity":
        s = conditions[key]["check_condition_value"]
        q = 0.0
        q_ = int(q)
        s_ = int(s)
    return q,s,q_,s_


def make_input_orca_inp(Z,R,n,q,s):

    max_iter_str = "%xtb XTBINPUTSTRING" +' "'+  "--iterations 5000" +'"' + " end\n"
    uhf_str = "%xtb XTBINPUTSTRING2 " + '"' +"--uhf "+str(s) +'"'+ "end\n"

    input_file = [
            "!SP XTB2\n",
            max_iter_str,
            uhf_str,
            "%scf maxiter 100 end\n",
            "%base "+'"'+path+"/"+str(n)+'"'+"\n",
            "*xyz "+str(q)+" "+str(s)+"\n"]
    end_part = ["*"]

    #write new struct
    STRINGS = []
    for i in range(len(Z)):
            st = (str(Z[i])+" "+str(R[i][0])+" "+str(R[i][1])+" "+str(R[i][2])+"\n")
            STRINGS.append(st)
    input_file.extend(STRINGS)
    input_file.extend(end_part)

    return input_file


def get_energy(mol,converter,model):

    #get the model prediction
    inputs = converter(mol)
    E = model(inputs)["energy"].detach().cpu().numpy().item()
    return E

def read_prop_file(prop_file):

    with open(prop_file) as f:
        data = f.readlines()
        for l in data:
            if "Total Energy" in l:
                e = float(l.split(" ")[-1].split("\n")[0])
    return e

results = {
    "Ag3+":{},
    "Ag3-":{},
    "singlet-CH2":{},
    "triplet-CH2":{}
}

def get_stats(q_,s_,data,key):
    splits = np.arange(1,2001,1)
    E = []
    for i in splits:
        if i.item() != 0:
            
            if (data.conn.get(i.item()+1).data["total_charge"].item() == q_ ) and (data.conn.get(i.item()+1).data["spin_multiplicity"].item() == s_):
                E.append(data.conn.get(i.item()+1).data["energy"])

    E = np.mean(np.concatenate(np.array(E))).item()
    print(f"{key} energy mean {E}")


for key in conditions.keys():

    #check if either spin or charge is the condition
    q,s,q_,s_ = get_conditons(conditions,key)

    #sampling grid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    grid_points = 25
    energies = np.zeros((grid_points,grid_points))
    ref_energies = np.zeros((grid_points,grid_points))
    ri = np.linspace(conditions[key]["r"][0],conditions[key]["r"][1],grid_points)
    ai = np.linspace(conditions[key]["alpha"][0],conditions[key]["alpha"][1],grid_points)

    #define start mol structure
    #mol = read(os.path.join(BASEPATH,conditions[key]["ref_mol_opt"]))
    db_path = BASEPATH+"/"+ conditions[key]["database"]
    data = ASEAtomsData(datapath = db_path)
    #get_stats(q,s,data,key)
    #indices to iterate over
    iteridx = list(product(range(grid_points),range(grid_points)))

    for n,idx in tqdm(enumerate(iteridx)):
        base = os.path.join(BASEPATH,conditions[key]["base"])
        # foler path
        path = os.path.join(base,str(n))
        # inp path
        inp_path = os.path.join(path,"input.inp") 
        # out path
        out_path = os.path.join(path,"out")
        # coords path
        coord_path = os.path.join(path,"coords.xyz")
        # prop path
        prop_path = os.path.join(path,str(n)+"_property.txt")
        # xyz path
        xyz_path = BASEPATH+"/"+conditions[key]["mol_spk"]

        #load the model and set to eval
        chk_point = BASEPATH+"/"+conditions[key]["model_file"]
        model = torch.load(chk_point,map_location=device)
        model.eval()


        if not os.path.exists(path):
            os.mkdir(path)

        #get new distance and angle
        r = ri[idx[0]]
        a = ai[idx[1]]

        #traj of displacement structs
        mol = read(xyz_path)

        #set new distance and angle
        mol.set_distance(0,1,r,mic=True,fix=0)
        mol.set_distance(0,2,r,mic=True,fix=0)
        mol.set_angle(2,0,1,a)

        Z = mol.get_atomic_numbers()
        R = mol.get_positions()

        input_file = make_input_orca_inp(Z,R,n,q_,s_)
        with open(inp_path,"w") as f:
            f.writelines(input_file)
        write(coord_path,mol)

        #init converter to get nhb list indices
        converter = spk.interfaces.AtomsConverter(
                neighbor_list=trn.ASENeighborList(cutoff=5.),
                additional_inputs={"total_charge":torch.tensor([q],device=device),"spin_multiplicity":torch.tensor([s],device=device)},device=device)



        run_orca(inp_path,out_path)
        ref_e = read_prop_file(prop_path) 
        pred_e = get_energy(mol,converter,model)

        energies[idx[0],idx[1]] = pred_e
        ref_energies[idx[0],idx[1]] = ref_e


    #store the results
    res = {"energies":energies,"ref_energies":ref_energies,"r":ri,"alpha":ai}
    results[key] = res

    print("Saving results")
    save_pes_path = os.path.join(BASEPATH,"PES_sampling_"+key+".pickle")
    #save_path = "/home/elron/phd/projects/google/experiments/debug_test_q_s_embedding/PES_sampling_"+key+".pickle"
    with open(save_pes_path,"wb") as f:
        pickle.dump(res,f)


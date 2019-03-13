import torch
from ase.db import connect
from schnetpack.ase_interface import MLPotential, Model


# path definitions
path_to_model = './../sacred_scripts/experiments/training/best_model'
path_to_db = './data/snippet.db'
# load model
model = torch.load(path_to_model)
# get example atom
conn = connect(path_to_db)
ats = conn.get_atoms(1)
# build calculator
w_model = Model(model=model, type='schnet', device='cpu')
calc = MLPotential(w_model)
# add calculator to atoms object
ats.set_calculator(calc)

#test
print('forces:', ats.get_forces())
print('total_energy', ats.get_total_energy())


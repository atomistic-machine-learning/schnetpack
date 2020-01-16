import torch
from ase.db import connect
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model


# path definitions
path_to_model = "tutorials/training/best_model"
path_to_db = "tutorials/data/snippet.db"
# load model
model = load_model(path_to_model)
# get example atom
conn = connect(path_to_db)
ats = conn.get_atoms(1)
# build calculator
calc = SpkCalculator(model, device="cpu", energy="energy", forces="forces")
# add calculator to atoms object
ats.set_calculator(calc)

# test
print("forces:", ats.get_forces())
print("total_energy", ats.get_total_energy())

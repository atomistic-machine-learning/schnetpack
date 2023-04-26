import torch
import random

from schnetpack.interfaces.ensemble_calculator import EnsembleCalculator,SimpleEnsembleAverage
from schnetpack.interfaces.ase_interface import AtomsConverter
import schnetpack as spk
from schnetpack.interfaces.batchwise_optimization import ASEBatchwiseLBFGS, BatchwiseCalculator

from ase import Atoms
from ase.io import read,Trajectory
from ase.visualize import view
from ase.geometry.analysis import get_rdf


'''
Test cases:

1. Batchwise Calculator plus Batchwise Optimization
2. EnsembleCalculator without uncertainity method plus Batchwise Optimization
3. EnsembleCalculator with uncertainity method plus Batchwise Optimization


'''


# set device and prepare model paths
device = torch.device("cuda")
path = "../../tests/testdata/md_ethanol.model"
xyz_path = "../../tests/testdata/md_ethanol.xyz"
models = [path] * 5

# define neighbor list
model = torch.load(models[0],map_location=device)
cutoff = model.representation.cutoff.item()
nbh_list=spk.transform.MatScipyNeighborList(cutoff=cutoff)

# build atoms converter
atoms_converter = AtomsConverter(
    neighbor_list=nbh_list,
    device=device,
)

# build calculator

#ensemble_average_strategy = SimpleEnsembleAverage(mode="batchwise")
ensemble_average_strategy = None

ensemble_calc = EnsembleCalculator(
    models,
    atoms_converter,
    device = device,
    energy_unit="kcal/mol",
    position_unit="Ang",
    ensemble_average_strategy = ensemble_average_strategy
    )


# build calculator
calculator = BatchwiseCalculator(
    model_file=models[0],
    atoms_converter=atoms_converter,
    device=device,
    energy_unit="kcal/mol",
    position_unit="Ang",
)


# create slightly distored ethanol mol
mol = read(xyz_path,index=":")
pos = mol[0].get_positions()
random.seed(42)
ats = []
for a in range(10):

    for n in range(pos.shape[0]):

        pos[n] = pos[n] * random.uniform(0.9,1.10)

    ats.append(Atoms(positions = pos, numbers = mol[0].get_atomic_numbers()))
    pos = mol[0].get_positions()


# Initialize optimizer
optimizer = ASEBatchwiseLBFGS(
    calculator=calculator,
    atoms=ats,
    trajectory="relax_traj",
)

# run optimization
optimizer.run(fmax=0.0005, steps=1000)


# get list of optimized structures and properties
opt_atoms, opt_props = optimizer.get_relaxation_results()

#positions = {str(n): opt_atoms[n].get_positions() for n in range(len(opt_atoms))}
#np.savez("batchwise_single",**positions)

for oatoms in opt_atoms:
    print(oatoms.get_positions())
    
print(opt_props)


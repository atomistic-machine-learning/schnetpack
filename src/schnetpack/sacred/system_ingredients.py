from sacred import Ingredient
from schnetpack.md.system import System


system_ingredient = Ingredient('system')


@system_ingredient.config
def config():
    """settings for the system ingredient"""
    n_replicas = 2


@system_ingredient.capture
def build_system(n_replicas, device, path_to_molecules):
    system = System(n_replicas, device)

    system.load_molecules_from_xyz(path_to_molecules)
    return system

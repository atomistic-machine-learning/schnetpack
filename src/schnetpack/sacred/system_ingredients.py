from sacred import Ingredient
from schnetpack.md.system import System
from schnetpack.sacred.initializer_ingredient import initializer_ing, \
    build_initializer


system_ingredient = Ingredient('system', ingredients=[initializer_ing])


@system_ingredient.config
def config():
    """configuration for the system ingredient"""
    n_replicas = 2


@system_ingredient.capture
def build_system(n_replicas, device, path_to_molecules):
    initializer_object = build_initializer()
    system = System(n_replicas, device, initializer=initializer_object)

    system.load_molecules_from_xyz(path_to_molecules)
    return system

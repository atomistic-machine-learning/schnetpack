from sacred import Ingredient
from schnetpack.md.system import System
from schnetpack.sacred.initializer_ingredient import initializer_ing, \
    build_initializer

system_ingredient = Ingredient('system', ingredients=[initializer_ing])


@system_ingredient.config
def config():
    """configuration for the system ingredient"""
    n_replicas = 1
    path_to_molecules = 'ethanol.xyz'


@system_ingredient.named_config
def ring_polymer():
    """configuration for the system ingredient"""
    n_replicas = 4
    path_to_molecules = 'ethanol.xyz'


@system_ingredient.capture
def build_system(_log, n_replicas, device, path_to_molecules):
    initializer_object = build_initializer()
    _log.info(f'Setting up system with {n_replicas} replicas')
    system = System(n_replicas, device, initializer=initializer_object)

    _log.info(f'Loading molecules from {path_to_molecules}...')
    system.load_molecules_from_xyz(path_to_molecules)
    _log.info(f'Found {system.n_molecules} molecules...')

    return system

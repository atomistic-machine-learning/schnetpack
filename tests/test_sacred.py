import os

import pytest

from sacred_scripts.run_md import md
from sacred_scripts.run_schnetpack import ex
from schnetpack.atomistic import Properties
from schnetpack.datasets.iso17 import ISO17


@pytest.fixture
def property_mapping():
    return {Properties.energy: ISO17.E, Properties.forces: ISO17.F}


@pytest.fixture
def properties(property_mapping):
    return [Properties.energy, Properties.forces]


def test_run_training(tmpdir, property_mapping, properties):
    """
    Test if training is working and logs are created.
    """
    dbpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'data/test_iso.db')

    ex.run(command_name='train',
           named_configs=['model.schnet'],
           config_updates={'experiment_dir': tmpdir,
                           'properties': properties,
                           'dataset.dbpath': dbpath,
                           'dataset.property_mapping': property_mapping,
                           'trainer.max_epochs': 4,
                           'batch_size': 2,
                           'trainer.logging_hooks': ['csv'],
                           'trainer.metrics': ['mae', 'rmse'],
                           'num_train': 4,
                           'num_val': 4,
                           })

    with open(os.path.join(tmpdir, 'training/log.csv'), 'r') as file:
        log = file.readlines()
    assert len(log[0].split(',')) == 8
    assert len(log) == 5


@pytest.fixture(params=[
    None,
    'thermostat.berendsen',
    'thermostat.langevin',
    'thermostat.gle',
    'thermostat.nhc',
    'thermostat.nhc_massive'
])
def md_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(params=[
    None,
    'system.ring_polymer'
])
def md_system(request):
    system = request.param
    return system


def test_run_md(md_thermostats, md_system, tmpdir):
    mol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data/test_molecule.xyz')
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'data/test_md_model.model')

    # Default test configs
    config_updates = {'experiment_dir': tmpdir,
                      'system.path_to_molecules': mol_path,
                      'calculator.model_path': model_path,
                      'simulation_steps': 2}

    named_configs = ['simulator.log_temperature',
                     'simulator.remove_com_motion']

    if md_thermostats is not None:
        named_configs.append(md_thermostats)

    if md_system is not None:
        named_configs.append(md_system)

    # Set input file path for GLE thermostat if used
    if md_thermostats == 'thermostat.gle':
        gle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data/test_gle_thermostat.txt')
        config_updates['thermostat.gle_file'] = gle_path

    md.run(command_name='simulate',
           named_configs=named_configs,
           config_updates=config_updates)


@pytest.fixture(params=[
    None,
    'thermostat.piglet',
    'thermostat.pile_local',
    'thermostat.pile_global',
    'thermostat.nhc_ring_polymer',
    'thermostat.nhc_ring_polymer_global'
])
def rpmd_thermostats(request):
    thermostat = request.param
    return thermostat


def test_run_rpmd(rpmd_thermostats, tmpdir):
    mol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data/test_molecule.xyz')
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'data/test_md_model.model')

    # Default test configs
    config_updates = {'experiment_dir': tmpdir,
                      'system.path_to_molecules': mol_path,
                      'calculator.model_path': model_path,
                      'simulation_steps': 2}

    named_configs = ['simulator.log_temperature',
                     'simulator.remove_com_motion',
                     'system.ring_polymer',
                     'integrator.ring_polymer',
                     'initializer.remove_com']

    if rpmd_thermostats is not None:
        named_configs.append(rpmd_thermostats)

    # Set input file path for GLE thermostat if used
    if rpmd_thermostats == 'thermostat.piglet':
        gle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data/test_piglet_thermostat.txt')
        config_updates['thermostat.gle_file'] = gle_path

    md.run(command_name='simulate',
           named_configs=named_configs,
           config_updates=config_updates)

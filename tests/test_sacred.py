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


# def test_run_md(tmpdir):
#     print('runmd', tmpdir)
#     mol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                             'data/test_molecule.xyz')
#     md.run(command_name='simulate',
#            named_configs=['thermostat.berendsen', 'simulator.base_hooks'],
#            config_updates={'experiment_dir': tmpdir,
#                            'path_to_molecules': mol_path,
#                            'simulation_steps': 10})

from sacred_scripts.run_schnetpack import ex
import tempfile
import shutil
import os
from schnetpack.atomistic import Properties
from schnetpack.datasets.qm9 import QM9
import pytest


tmpdir = tempfile.mkdtemp('gdb9')


@pytest.fixture
def property_mapping():
    return {Properties.energy: QM9.U0, Properties.dipole_moment: QM9.mu,
            Properties.iso_polarizability: QM9.alpha}


@pytest.fixture
def properties(property_mapping):
    return [Properties.energy, Properties.dipole_moment,
            Properties.iso_polarizability]


def test_run(property_mapping, properties):
    """
    Test if training is working and logs are created.
    """
    dbpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'data/test_qm9.db')
    ex.run(command_name='train',
           named_configs=['model.schnet'],
           config_updates={'modeldir': tmpdir,
                           'properties': properties,
                           'dataset.dbpath': dbpath,
                           'dataset.property_mapping': property_mapping,
                           'trainer.max_epochs': 4,
                           'batch_size': 2,
                           'trainer.logging_hooks': ['csv'],
                           'trainer.metrics': ['mae', 'rmse']
                           })

    with open(os.path.join(tmpdir, 'log.csv'), 'r') as file:
        log = file.readlines()
    assert len(log[0].split(',')) == 10
    assert len(log) == 5


def teardown_module():
    shutil.rmtree(tmpdir)

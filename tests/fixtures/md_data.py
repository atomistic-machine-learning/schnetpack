import os
import tempfile

import pytest

__all__ = [
    'simulation_dir',
    'model_path',
    'molecule_path',
    'gle_path',
    'piglet_path',
    'md_config'
]


@pytest.fixture(scope="module")
def simulation_dir():
    return tempfile.mkdtemp()


@pytest.fixture(scope="module")
def model_path():
    model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test_md_model.model")
    return model


@pytest.fixture(scope="module")
def molecule_path():
    mol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test_molecule.xyz")
    return mol_path


@pytest.fixture(scope="module")
def gle_path():
    gle_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test_gle_themostat.txt")
    return gle_file


@pytest.fixture(scope="module")
def piglet_path():
    piglet_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test_piglet_themostat.txt")
    return piglet_file


@pytest.fixture(scope="module")
def md_config():
    config = {
        'device': 'cuda',
        'simulation_dir': 'SIMULATION_DIR',
        'seed': 662524648,
        'overwrite': True,
        'calculator': {
            'type': 'schnet',
            'model_file': 'MODEL_PATH',
            'required_properties': ['energy', 'forces'],
            'force_handle': 'forces'
        },
        'system': {
            'molecule_file': 'MOLECULE_PATH',
            'n_replicas': 1,
            'initializer': {
                'type': 'maxwell-boltzmann',
                'temperature': 300,
                'remove_translation': True,
                'remove_rotation': False
            }
        },
        'dynamics': {
            'n_steps': 2,
            'integrator': {
                'type': 'verlet',
                'time_step': 0.5,
            },
            'remove_com_motion': {
                'every_n_steps': 100,
                'remove_rotation': True
            }
        },
        'logging': {
            'file_logger': {
                'buffer_size': 50,
                'streams': ['molecules', 'properties', 'dynamic']
            },
            'temperature_logger': 50,
            'write_checkpoints': 100
        }
    }
    return config

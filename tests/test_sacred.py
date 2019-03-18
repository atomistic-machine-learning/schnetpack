import os
import pytest
import tempfile

from sacred_scripts.spk_md import md
from sacred_scripts.spk_train import ex
from schnetpack.atomistic import Properties
from schnetpack.datasets.iso17 import ISO17


@pytest.fixture(scope='module')
def property_mapping():
    return {Properties.energy: ISO17.E, Properties.forces: ISO17.F}


@pytest.fixture(scope='module')
def properties(property_mapping):
    return [Properties.energy, Properties.forces]


@pytest.fixture(scope='module')
def tmpdir():
    return tempfile.mkdtemp()


@pytest.fixture(scope='module')
def simulation_dir(tmpdir):
    return os.path.join(tmpdir, 'simulate')


@pytest.fixture(scope='module')
def training_dir(tmpdir):
    return os.path.join(tmpdir, 'train')


@pytest.fixture(params=[None, 'thermostat.berendsen', 'thermostat.langevin',
                        'thermostat.gle', 'thermostat.nhc',
                        'thermostat.nhc_massive'])
def md_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(params=[None, 'thermostat.piglet', 'thermostat.pile_local',
                        'thermostat.pile_global',
                        'thermostat.nhc_ring_polymer',
                        'thermostat.nhc_ring_polymer_global'])
def rpmd_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(params=[None, 'system.ring_polymer'])
def md_system(request):
    system = request.param
    return system


@pytest.fixture(scope='module')
def generate_model(training_dir, properties, property_mapping):
    dbpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'data/test_iso.db')
    ex.run(command_name='train',
           named_configs=[],
           config_updates={'model_dir': training_dir,
                           'properties': properties,
                           'dataset.dbpath': dbpath,
                           'dataset.property_mapping': property_mapping,
                           'dataloader.batch_size': 2,
                           'metrics.names': ['mae', 'rmse'],
                           'dataloader.num_train': 4,
                           'dataloader.num_val': 4,
                           'early_stopping.max_epochs': 4
                           })


class TestSacred:

    def test_run_training(self, training_dir, generate_model):
        """
        Test if training is working and logs are created.
        """
        with open(os.path.join(training_dir, 'log.csv'), 'r') as file:
            log = file.readlines()
        assert len(log[0].split(',')) == 8
        assert len(log) == 5

    def test_run_md(self, training_dir, simulation_dir, md_thermostats,
                    md_system, generate_model):
        mol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data/test_molecule.xyz')
        model_path = os.path.join(training_dir, 'best_model')

        # Default test configs
        config_updates = {'simulation_dir': simulation_dir,
                          'system.path_to_molecules': mol_path,
                          'calculator.model_path': model_path,
                          'calculator.required_properties': ['energy',
                                                             'forces'],
                          'calculator.force_handle': 'forces',
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

    def test_run_rpmd(self, training_dir, simulation_dir, rpmd_thermostats,
                      generate_model):
        mol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data/test_molecule.xyz')
        model_path = os.path.join(training_dir, 'best_model')

        # Default test configs
        config_updates = {'simulation_dir': simulation_dir,
                          'system.path_to_molecules': mol_path,
                          'calculator.model_path': model_path,
                          'calculator.force_handle': 'forces',
                          'calculator.required_properties': ['energy', 'forces'],
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

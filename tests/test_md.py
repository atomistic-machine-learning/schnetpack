import pytest
from schnetpack.md.parsers.md_setup import MDSimulation
from .fixtures import *


@pytest.fixture(
    params=[
        None,
        "berendsen",
        "langevin",
        "nhc",
        "nhc-massive",
        "gle"
    ]
)
def md_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(
    params=[
        None,
        "pile-l",
        "pile-g",
        "trpmd",
        "piglet",
        "pi-nhc-l",
        "pi-nhc-g"
    ]
)
def rpmd_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(scope='module')
def get_thermostat_defaults(thermostat):
    thermostat_config = {
        'type': thermostat,
        'temperature': 300
    }

    if thermostat == 'piglet':
        thermostat_config['gle_input'] = None
    elif thermostat == 'gle':
        thermostat_config['gle_input'] = None
    elif thermostat == 'trpmd':
        thermostat_config['damping'] = 0.5
    else:
        thermostat_config['time_constant'] = 100.0

    return thermostat_config


@pytest.fixture(scope='module')
def md_integrator():
    return {
        'integrator': {
            'type': 'verlet',
            'time_step': 0.5,
        },
    }


@pytest.fixture(scope='module')
def rpmd_integrator():
    return {
        'integrator': {
            'type': 'ring_polymer',
            'time_step': 0.5,
            'temperature': 300.0,
        },
    }


def setup_basic_md(md_config, simulation_dir, molecule_file, model_path):
    md_config['simulation_dir'] = simulation_dir
    md_config['calculator']['model_file'] = model_path
    md_config['system']['molecule_file'] = molecule_file
    return md_config


@pytest.fixture(scope='module')
def setup_md(md_config, md_integrator, md_thermostats):
    # Setup system replicas
    md_config['system']['n_replicas'] = 1

    # Set integrator
    md_config['dynamics'].update(md_integrator)

    # Set thermostat
    if md_thermostats is not None:
        thermostat_config = get_thermostat_defaults(md_thermostats)
        md_config['dynamics'].update(thermostat_config)

    return md_config


@pytest.fixture(scope='module')
def setup_rpmd(md_config, rpmd_integrator, rpmd_thermostats):
    # Setup system replicas
    md_config['system']['n_replicas'] = 4

    # Set integrator
    md_config['dynamics'].update(rpmd_integrator)

    # Set thermostat
    if md_thermostats is not None:
        thermostat_config = get_thermostat_defaults(rpmd_thermostats)
        md_config['dynamics'].update(thermostat_config)

    return md_config


class TestSacred:

    @pytest.mark.filterwarnings("ignore")
    def test_run_md(self, md_config, simulation_dir, molecule_path, model_path, md_thermostats):
        md_config = setup_basic_md(md_config, simulation_dir, molecule_path, model_path)
        md_config = setup_md(md_config, md_integrator, md_thermostats)

        md = MDSimulation(md_config)
        md.save_config()
        md.run()

    @pytest.mark.filterwarnings("ignore")
    def test_run_rpmd(self, md_config, simulation_dir, molecule_path, model_path, rpmd_thermostats):
        md_config = setup_basic_md(md_config, simulation_dir, molecule_path, model_path)
        md_config = setup_md(md_config, rpmd_integrator, rpmd_thermostats)

        md = MDSimulation(md_config)
        md.save_config()
        md.run()

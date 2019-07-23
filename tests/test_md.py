import os
import tempfile

import pytest
from schnetpack.md.parsers.md_setup import MDSimulation

import logging


@pytest.fixture(scope="module")
def simulation_dir():
    tempdir = tempfile.mkdtemp()
    return tempdir


@pytest.fixture(scope="module")
def model_path():
    model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data/test_md_model.model"
    )
    return model


@pytest.fixture(scope="module")
def molecule_path():
    mol_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data/test_molecule.xyz"
    )
    return mol_path


@pytest.fixture
def md_config():
    config = {
        "device": "cpu",
        "simulation_dir": "SIMULATION_DIR",
        "seed": 662524648,
        "overwrite": True,
        "calculator": {
            "type": "schnet",
            "model_file": "MODEL_PATH",
            "required_properties": ["energy", "forces"],
            "force_handle": "forces",
        },
        "system": {
            "molecule_file": "MOLECULE_PATH",
            "n_replicas": 1,
            "initializer": {
                "type": "maxwell-boltzmann",
                "temperature": 300,
                "remove_translation": True,
                "remove_rotation": False,
            },
        },
        "dynamics": {
            "n_steps": 2,
            "integrator": {"type": "verlet", "time_step": 0.5},
            "remove_com_motion": {"every_n_steps": 100, "remove_rotation": True},
        },
        "logging": {
            "file_logger": {
                "buffer_size": 50,
                "streams": ["molecules", "properties", "dynamic"],
            },
            "temperature_logger": 50,
            "write_checkpoints": 100,
        },
    }
    return config


@pytest.fixture(params=[None, "berendsen", "langevin", "nhc", "nhc-massive", "gle"])
def md_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(
    params=[None, "pile-l", "pile-g", "trpmd", "piglet", "pi-nhc-l", "pi-nhc-g"]
)
def rpmd_thermostats(request):
    thermostat = request.param
    return thermostat


def get_thermostat_defaults(thermostat):
    thermostat_config = {"type": thermostat, "temperature": 300}

    if thermostat == "gle":
        thermostat_config["gle_input"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data/test_gle_thermostat.txt"
        )
    elif thermostat == "piglet":
        thermostat_config["gle_input"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data/test_piglet_thermostat.txt",
        )
    elif thermostat == "trpmd":
        thermostat_config["damping"] = 0.5
    else:
        thermostat_config["time_constant"] = 100.0

    return thermostat_config


@pytest.fixture(scope="module")
def md_integrator():
    return {"type": "verlet", "time_step": 0.5}


@pytest.fixture(scope="module")
def rpmd_integrator():
    return {"type": "ring_polymer", "time_step": 0.2, "temperature": 300.0}


def setup_basic_md(md_config, simulation_dir, molecule_file, model_path):
    md_config["simulation_dir"] = simulation_dir
    md_config["calculator"]["model_file"] = model_path
    md_config["system"]["molecule_file"] = molecule_file
    return md_config


def setup_md(md_config, md_integrator, md_thermostats):
    # Setup system replicas
    md_config["system"]["n_replicas"] = 1

    # Set integrator
    md_config["dynamics"]["integrator"] = md_integrator

    # Set thermostat
    if md_thermostats is not None:
        thermostat_config = get_thermostat_defaults(md_thermostats)
        md_config["dynamics"]["thermostat"] = thermostat_config

    return md_config


def setup_rpmd(md_config, rpmd_integrator, rpmd_thermostats):
    # Setup system replicas
    md_config["system"]["n_replicas"] = 4

    # Set integrator
    md_config["dynamics"]["integrator"] = rpmd_integrator

    # Set thermostat
    if rpmd_thermostats is not None:
        thermostat_config = get_thermostat_defaults(rpmd_thermostats)
        md_config["dynamics"]["thermostat"] = thermostat_config

    return md_config


class TestSacred:
    @pytest.mark.filterwarnings("ignore")
    def test_run_md(
        self,
        md_config,
        simulation_dir,
        molecule_path,
        model_path,
        md_thermostats,
        md_integrator,
    ):
        md_config = setup_basic_md(md_config, simulation_dir, molecule_path, model_path)
        md_config = setup_md(md_config, md_integrator, md_thermostats)

        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test loading of system state
        md_config["dynamics"]["load_system_state"] = os.path.join(
            simulation_dir, "checkpoint.chk"
        )
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test restart
        md_config["overwrite"] = False
        md_config["dynamics"]["restart"] = os.path.join(
            simulation_dir, "checkpoint.chk"
        )
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

    @pytest.mark.filterwarnings("ignore")
    def test_run_rpmd(
        self,
        md_config,
        simulation_dir,
        molecule_path,
        model_path,
        rpmd_thermostats,
        rpmd_integrator,
    ):
        md_config = setup_basic_md(md_config, simulation_dir, molecule_path, model_path)
        md_config = setup_rpmd(md_config, rpmd_integrator, rpmd_thermostats)
        logging.info(md_thermostats)

        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test loading of system state
        md_config["dynamics"]["load_system_state"] = os.path.join(
            simulation_dir, "checkpoint.chk"
        )
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test restart
        md_config["overwrite"] = False
        md_config["dynamics"]["restart"] = os.path.join(
            simulation_dir, "checkpoint.chk"
        )
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

import os
import logging
import pytest

from schnetpack.md.parsers.md_setup import MDSimulation
from tests.fixtures import *


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
        sim_dir,
        molecule_path,
        md_model_path,
        md_thermostats,
        md_integrator,
    ):
        md_config = setup_basic_md(md_config, sim_dir, molecule_path, md_model_path)
        md_config = setup_md(md_config, md_integrator, md_thermostats)

        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test loading of system state
        md_config["dynamics"]["load_system_state"] = os.path.join(
            sim_dir, "checkpoint.chk"
        )
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test restart
        md_config["overwrite"] = False
        md_config["dynamics"]["restart"] = os.path.join(sim_dir, "checkpoint.chk")
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

    @pytest.mark.filterwarnings("ignore")
    def test_run_rpmd(
        self,
        md_config,
        sim_dir,
        molecule_path,
        md_model_path,
        rpmd_thermostats,
        rpmd_integrator,
    ):
        md_config = setup_basic_md(md_config, sim_dir, molecule_path, md_model_path)
        md_config = setup_rpmd(md_config, rpmd_integrator, rpmd_thermostats)
        logging.info(md_thermostats)

        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test loading of system state
        md_config["dynamics"]["load_system_state"] = os.path.join(
            sim_dir, "checkpoint.chk"
        )
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

        # Test restart
        md_config["overwrite"] = False
        md_config["dynamics"]["restart"] = os.path.join(sim_dir, "checkpoint.chk")
        md = MDSimulation(md_config)
        md.save_config()
        md.run()

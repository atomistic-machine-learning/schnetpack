from sacred import Experiment
import os

try:
    import oyaml as yaml
except ImportError:
    import yaml

from shutil import rmtree

from schnetpack.sacred_ingredients.calculator_ingredients import (
    calculator_ingredient,
    build_calculator,
)
from schnetpack.sacred_ingredients.simulator_ingredients import (
    simulator_ingredient,
    build_simulator,
)
from schnetpack.sacred_ingredients.integrator_ingredients import (
    integrator_ingredient,
    build_integrator,
)
from schnetpack.sacred_ingredients.system_ingredients import (
    system_ingredient,
    build_system,
)
from schnetpack.sacred_ingredients.thermostat_ingredients import (
    thermostat_ingredient,
    build_thermostat,
)

md = Experiment(
    "md",
    ingredients=[
        simulator_ingredient,
        calculator_ingredient,
        integrator_ingredient,
        system_ingredient,
        thermostat_ingredient,
    ],
)

SETUP_STRING_WIDTH = 30
SETUP_STRING = "\n\n{:s}\n{:s}\n{:s}".format(
    SETUP_STRING_WIDTH * "=", f"{{:^{SETUP_STRING_WIDTH}s}}", SETUP_STRING_WIDTH * "="
)


@md.config
def config():
    """configuration for the simulation experiment"""
    simulation_dir = "experiment"
    simulation_steps = 1000
    device = "cpu"
    overwrite = True


@md.capture
def save_system_config(_config, simulation_dir):
    """
    Save the configuration to the model directory.

    Args:
        _config (dict): configuration of the experiment
        simulation_dir (str): path to the simulation directory

    """
    with open(os.path.join(simulation_dir, "config.yaml"), "w") as f:
        yaml.dump(_config, f, default_flow_style=False)


@md.capture
def setup_simulation(_log, simulation_dir, device):
    _log.info(SETUP_STRING.format("CALCULATOR SETUP"))
    calculator = build_calculator(device=device)
    _log.info(SETUP_STRING.format("SYSTEM SETUP"))
    system = build_system(device=device)
    _log.info(SETUP_STRING.format("INTEGRATOR SETUP"))
    integrator = build_integrator(n_beads=system.n_replicas, device=device)
    _log.info(SETUP_STRING.format("THERMOSTAT SETUP"))
    thermostat = build_thermostat()
    _log.info(SETUP_STRING.format("SIMULATOR SETUP"))
    simulator = build_simulator(
        system=system,
        integrator_object=integrator,
        calculator_object=calculator,
        simulation_dir=simulation_dir,
        thermostat_object=thermostat,
    )
    return simulator


@md.capture
def create_dirs(_log, simulation_dir, overwrite):
    """
    Create the directory for the experiment.

    Args:
        _log:
        experiment_dir (str): path to the experiment directory
        overwrite (bool): overwrites the model directory if True

    """

    _log.info("Create model directory")
    if simulation_dir is None:
        raise ValueError("Config `simulation_dir` has to be set!")

    if os.path.exists(simulation_dir) and not overwrite:
        raise ValueError(
            "Model directory already exists (set overwrite flag?):", simulation_dir
        )

    if os.path.exists(simulation_dir) and overwrite:
        rmtree(simulation_dir)

    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)


@md.command
def simulate(simulation_dir, simulation_steps):
    create_dirs()
    save_system_config()
    simulator = setup_simulation(simulation_dir=simulation_dir)
    simulator.simulate(simulation_steps)


@md.command
def save_config(_log, _config, simulation_dir):
    file_name = f"{simulation_dir}_config.yaml"
    with open(file_name, "w") as f:
        yaml.dump(_config, f, default_flow_style=False)
    _log.info(f"Stored config to {file_name}")


@md.automain
def main(_log):
    save_config()
    _log.info('To run simulation call script with "simulate with <config file>"')

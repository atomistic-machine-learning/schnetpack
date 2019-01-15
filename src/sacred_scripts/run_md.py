from sacred import Experiment
import os
import torch
import yaml

from schnetpack.sacred.calculator_ingredients import (calculator_ingradient,
                                                      build_calculator)
from schnetpack.sacred.simulator_ingredients import (simulator_ingredient,
                                                     build_simulator)
from schnetpack.sacred.model_ingredients import (model_ingredient, build_model)
from schnetpack.sacred.integrator_ingredients import (integrator_ingredient,
                                                      build_integrator)
from schnetpack.sacred.system_ingredients import (system_ingredient,
                                                  build_system)
from schnetpack.sacred.thermostat_ingredients import thermostat_ingredient, \
    build_thermostat


md = Experiment('md', ingredients=[simulator_ingredient, calculator_ingradient,
                                   integrator_ingredient, model_ingredient,
                                   system_ingredient, thermostat_ingredient])


@md.config
def config():
    modeldir = './runs/models'
    simulation_steps = 1000
    device = 'cpu'
    path_to_molecules = './runs/data/ethanol.xyz'


@md.capture
def save_config(_config, modeldir):
    """
    Save the configuration to the model directory.

    Args:
        _config (dict): configuration of the experiment
        modeldir (str): path to the model directory

    """
    with open(os.path.join(modeldir, 'simulation_config.yaml'), 'w') as f:
        yaml.dump(_config, f, default_flow_style=False)


@md.capture
def get_model_config(modeldir):
    with open(os.path.join(modeldir, 'model_config.yaml')) as f:
        model_config = yaml.load(f)
    return model_config


@md.capture
def get_state_dict(modeldir):
    state_dict = torch.load(os.path.join(modeldir, 'best_model'))
    return state_dict


@md.capture
def load_model(modeldir):
    return torch.load(os.path.join(modeldir, 'best_model'))


@md.capture
def setup_simulation(modeldir, device, path_to_molecules):
    model = load_model(modeldir)
    calculator = build_calculator(model=model)
    integrator = build_integrator()
    system = build_system(device=device, path_to_molecules=path_to_molecules)
    thermostat = build_thermostat()
    simulator = build_simulator(system=system,
                                integrator_object=integrator,
                                calculator_object=calculator,
                                modeldir=modeldir, thermostat_object=thermostat)
    return simulator


@md.command
def simulate(modeldir, simulation_steps):
    save_config()
    simulator = setup_simulation(modeldir=modeldir)
    simulator.simulate(simulation_steps)


@md.automain
def main():
    print(md.config)



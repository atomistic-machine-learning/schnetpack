import os
from sacred import Ingredient

from schnetpack.simulate.simulator import Simulator
from schnetpack.simulate.hooks import *
from schnetpack.simulate.thermostats import *


simulator_ingredient = Ingredient('simulator')


@simulator_ingredient.config
def config():
    """settings for the simulator ingredient"""
    logging_hooks = []
    data_streams = []
    step = 0
    log_every_n_steps = 100
    checkpoint_every_n_steps = 1000


@simulator_ingredient.named_config
def base_hooks():
    """default settings for logging hooks"""
    logging_hooks = ['file_logger', 'checkpoint_logger']
    buffer_size = 1
    data_streams = ['molecule_stream', 'property_stream']
    restart = False


@simulator_ingredient.capture
def build_simulator(system, integrator_object, calculator_object,
                    logging_hooks, step, thermostat_object, simulation_dir):
    hook_objects = [thermostat_object] if thermostat_object else []
    hook_objects += build_logging_hooks(logging_hooks, simulation_dir)
    return Simulator(system=system, integrator=integrator_object,
                     calculator=calculator_object, simulator_hooks=hook_objects,
                     step=step)


@simulator_ingredient.capture
def build_logging_hooks(simulator_hooks, simulation_dir):
    hook_objects = []
    for hook in simulator_hooks:
        if hook == 'file_logger':
            hook_objects.append(get_file_logger(simulation_dir=simulation_dir))
        elif hook == 'checkpoint_logger':
            hook_objects.append(get_checkpoint_logger(simulation_dir=simulation_dir))
        elif hook == 'remove_com_motion':
            hook_objects.append(get_remove_com_motion_logger())
        elif hook == 'bias_potential':
            hook_objects.append(get_bias_potential())
        elif hook == 'tensorboard_logger':
            hook_objects.append(
                get_tensorboard_logger(simulation_dir=simulation_dir))
        elif hook == 'temperature_logger':
            hook_objects.append(
                get_temperature_logger(simulation_dir=simulation_dir))
        else:
            raise NotImplementedError
    return hook_objects


@simulator_ingredient.capture
def get_tensorboard_logger(simulation_dir, log_every_n_steps):
    log_file = os.path.join(simulation_dir, 'tensorboard_log')
    return TensorboardLogger(log_file=log_file, every_n_steps=log_every_n_steps)


@simulator_ingredient.capture
def get_temperature_logger(simulation_dir, log_every_n_steps):
    log_file = os.path.join(simulation_dir, 'temperature_log')
    return TemperatureLogger(log_file=log_file, every_n_steps=log_every_n_steps)


@simulator_ingredient.capture
def get_bias_potential():
    raise NotImplementedError


@simulator_ingredient.capture
def get_remove_com_motion_logger(log_every_n_steps):
    return RemoveCOMMotion(every_n_steps=log_every_n_steps)


@simulator_ingredient.capture
def get_file_logger(simulation_dir, buffer_size, data_streams, restart):
    data_stream_classes = build_datastreams(data_streams)
    log_file = os.path.join(simulation_dir, 'log')
    return FileLogger(log_file, buffer_size, data_stream_classes, restart)


@simulator_ingredient.capture
def get_checkpoint_logger(simulation_dir, checkpoint_every_n_steps=1000):
    checkpoint_file = os.path.join(simulation_dir, 'checkpoint_file')
    return Checkpoint(checkpoint_file=checkpoint_file,
                      every_n_steps=checkpoint_every_n_steps)


@simulator_ingredient.capture
def build_datastreams(datastreams):
    stream_classes = []
    for datastream in datastreams:
        if datastream == 'molecule_stream':
            stream_classes.append(MoleculeStream)
        elif datastream == 'property_stream':
            stream_classes.append(PropertyStream)
        else:
            raise NotImplementedError
    return stream_classes

import os
import torch
from sacred import Ingredient

from schnetpack.simulate.simulator import Simulator
from schnetpack.simulate.hooks import *
from schnetpack.simulate.thermostats import *

simulator_ingredient = Ingredient("simulator")

DEFAULT_HOOKS = {
    "file_logger": {
        "every_n_steps": 1,
        "buffer_size": 100,
        "data_streams": ["molecule_stream", "property_stream"],
        "stream_options": {"log_properties": "all"},
    },
    "checkpoint_logger": {"every_n_steps": 100},
}


@simulator_ingredient.config
def config():
    """configuration for the simulator ingredient"""
    simulator_hooks = DEFAULT_HOOKS
    restart = False
    load_system_state = False


@simulator_ingredient.named_config
def log_temperature():
    simulator_hooks = DEFAULT_HOOKS
    simulator_hooks["temperature_logger"] = {"every_n_steps": 100}


@simulator_ingredient.named_config
def remove_com_motion():
    simulator_hooks = DEFAULT_HOOKS
    simulator_hooks["remove_com_motion"] = {
        "every_n_steps": 100,
        "remove_rotation": True,
    }


@simulator_ingredient.capture
def build_simulator(
    _log,
    system,
    integrator_object,
    calculator_object,
    simulator_hooks,
    thermostat_object,
    simulation_dir,
    restart,
    load_system_state,
):
    hook_objects = [thermostat_object] if thermostat_object else []
    hook_objects += build_simulation_hooks(simulator_hooks, simulation_dir)

    simulator = Simulator(
        system=system,
        integrator=integrator_object,
        calculator=calculator_object,
        simulator_hooks=hook_objects,
    )

    # If requested, read restart data
    if restart:
        state_dict = torch.load(restart)
        simulator.restart(state_dict, soft=False)
        _log.info(f"Restarting simulation from {restart}...")
    elif load_system_state:
        state_dict = torch.load(load_system_state)
        simulator.load_system_state(state_dict)
        _log.info(f"Loaded system state from {load_system_state}...")

    return simulator


@simulator_ingredient.capture
def build_simulation_hooks(simulator_hooks, simulation_dir):
    hook_objects = []
    for hook_name, hook_options in simulator_hooks.items():
        if hook_name == "file_logger":
            # Sets a default if not given explicitly
            if "stream_options" not in hook_options:
                hook_options["stream_options"] = {}
            hook_objects.append(
                get_file_logger(
                    simulation_dir=simulation_dir,
                    data_streams=hook_options["data_streams"],
                    buffer_size=hook_options["buffer_size"],
                    every_n_steps=hook_options["every_n_steps"],
                    stream_options=hook_options["stream_options"],
                )
            )
        elif hook_name == "checkpoint_logger":
            hook_objects.append(
                get_checkpoint_logger(
                    simulation_dir=simulation_dir,
                    every_n_steps=hook_options["every_n_steps"],
                )
            )
        elif hook_name == "tensorboard_logger":
            hook_objects.append(
                get_tensorboard_logger(
                    simulation_dir=simulation_dir,
                    every_n_steps=hook_options["every_n_steps"],
                )
            )
        elif hook_name == "temperature_logger":
            hook_objects.append(
                get_temperature_logger(
                    simulation_dir=simulation_dir,
                    every_n_steps=hook_options["every_n_steps"],
                )
            )
        elif hook_name == "remove_com_motion":
            hook_objects.append(
                get_remove_com_motion(
                    every_n_steps=hook_options["every_n_steps"],
                    remove_rotation=hook_options["remove_rotation"],
                )
            )
        elif hook_name == "bias_potential":
            hook_objects.append(get_bias_potential())
        else:
            raise NotImplementedError

    return hook_objects


@simulator_ingredient.capture
def get_tensorboard_logger(_log, simulation_dir, every_n_steps):
    log_file = os.path.join(simulation_dir, "tensorboard_log")
    return TensorboardLogger(log_file=log_file, every_n_steps=every_n_steps)


@simulator_ingredient.capture
def get_temperature_logger(_log, simulation_dir, every_n_steps):
    log_file = os.path.join(simulation_dir, "temperature_log")
    _log.info(f"Logging temperature to {log_file} every {every_n_steps} steps.")
    return TemperatureLogger(log_file=log_file, every_n_steps=every_n_steps)


@simulator_ingredient.capture
def get_checkpoint_logger(_log, simulation_dir, every_n_steps=1000):
    checkpoint_file = os.path.join(simulation_dir, "checkpoint_file.chk")
    _log.info(f"Writing checkpoint file {checkpoint_file} every {every_n_steps} steps.")
    return Checkpoint(checkpoint_file=checkpoint_file, every_n_steps=every_n_steps)


@simulator_ingredient.capture
def get_remove_com_motion(_log, every_n_steps, remove_rotation=False):
    _log.info(
        f'Removing center of mass translation{("", " and rotation")[remove_rotation]}'
        f" every {every_n_steps} steps."
    )
    return RemoveCOMMotion(every_n_steps=every_n_steps, remove_rotation=remove_rotation)


@simulator_ingredient.capture
def get_bias_potential():
    raise NotImplementedError


@simulator_ingredient.capture
def get_file_logger(
    _log,
    restart,
    simulation_dir,
    data_streams,
    buffer_size,
    every_n_steps,
    stream_options,
):
    data_stream_classes = build_datastreams(data_streams, stream_options)
    log_file = os.path.join(simulation_dir, "simulation.hdf5")
    _log.info(
        f'Writing data streams {", ".join(data_streams)} to {log_file} every {every_n_steps} steps.'
    )
    return FileLogger(
        log_file,
        buffer_size,
        data_stream_classes,
        restart=restart,
        every_n_steps=every_n_steps,
    )


@simulator_ingredient.capture
def build_datastreams(datastreams, stream_options):
    # TODO: Options should be given here
    stream_classes = []

    for datastream in datastreams:

        if datastream == "molecule_stream":
            stream_classes.append(MoleculeStream())

        elif datastream == "property_stream":
            # If present, check for additional options
            if "log_properties" in stream_options:
                log_properties = stream_options["log_properties"]
                # Map all to default
                if log_properties == "all":
                    log_properties = None
                elif type(log_properties) is not list:
                    raise TypeError("Expected a list of properties in stream_options.")
                stream_classes.append(PropertyStream(target_properties=log_properties))
            else:
                stream_classes.append(PropertyStream())
        else:
            raise NotImplementedError
    return stream_classes

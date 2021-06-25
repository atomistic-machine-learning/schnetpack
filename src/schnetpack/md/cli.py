import logging
import os
from shutil import rmtree
import uuid
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

import schnetpack.md
from schnetpack.utils import str2class, int2precision
from schnetpack.utils.script import print_config
from schnetpack.md.utils import (
    config_alias,
    get_alias,
    get_npt_integrator,
    is_rpmd_integrator,
    set_random_seed,
)

from ase.io import read

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))


class MDSetupError(Exception):
    pass


@hydra.main(config_path="md_configs", config_name="config")
def simulate(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(
        """
           _____      __    _   __     __  ____             __    __  __    ___
          / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__ |  \/  |  |   \\
          \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/ | |\/| |  | |) |
         ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<    |_|__|_|  |___/ 
        /____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|  _|""  ""|_|""  ""|
                                                                "`-0--0-'"`-0--0-'
        """
    )

    # Confound hydra
    os.chdir(hydra.utils.get_original_cwd())

    # Load custom config and use to update defaults
    if "load_config" in config:
        config_path = hydra.utils.to_absolute_path(config.load_config)
        logging.info("Loading config from {:s}".format(config_path))
        loaded_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, loaded_config)

    if config.get("print_config"):
        print_config(config, resolve=True)

    # ===========================================
    #   Initialize seed and simulation directory
    # ===========================================

    # Get device and precision
    device = config.device
    precision = int2precision(config.precision)

    # Set up random seed
    set_random_seed(config.seed)

    # Set up directories
    log.info("Setting up simulation directory...")
    if os.path.exists(config.simulation_dir) and not config.overwrite:
        log.warning(
            "Simulation directory {:s} already exists (set overwrite flag?)".format(
                config.simulation_dir
            )
        )
    elif os.path.exists(config.simulation_dir) and config.overwrite:
        rmtree(config.simulation_dir)
        os.makedirs(config.simulation_dir)
    else:
        os.makedirs(config.simulation_dir)

    # ===========================================
    #   Initialize the system
    # ===========================================

    log.info("Setting up system...")
    system = schnetpack.md.System()
    if config.system.load_system_state is not None:
        state_dict = torch.load(
            hydra.utils.to_absolute_path(config.system.load_system_state)
        )
        system.load_system_state(state_dict["system"])
        log.info(
            "Loaded previous system state from {:s}".format(
                config.system.load_system_state
            )
        )
    else:
        molecules = read(
            hydra.utils.to_absolute_path(config.system.molecule_file), index=":"
        )
        log.info(
            "Found {:d} molecule{:s}".format(
                len(molecules), ("", "s")[len(molecules) > 1]
            )
        )
        log.info(
            "Using {:d} replica{:s}".format(
                config.system.n_replicas, ("", "s")[config.system.n_replicas > 1]
            )
        )
        system.load_molecules(
            molecules=molecules,
            n_replicas=config.system.n_replicas,
            position_unit_input=config.system.position_unit_input,
            mass_unit_input=config.system.mass_unit_input,
        )
        if config.system.initializer is not None:
            log.info("Preparing initial conditions...")
            config.system.initializer = config_alias(config.system.initializer)
            initializer = hydra.utils.instantiate(config.system.initializer)
            initializer.initialize_system(system)

    # ===========================================
    #   Set up the calculator
    # ===========================================

    # TODO: check paths and default dir
    config.calculator = config_alias(config.calculator)
    # Change everything to dict for convenience
    calculator_config = dict(config.calculator)

    log.info("Setting up {:s} calculator...".format(calculator_config["_target_"]))

    # Check for neighbor lists
    if "neighbor_list" in calculator_config:
        log.info(
            "Using {:s} neighbor list...".format(calculator_config["neighbor_list"])
        )
        calculator_config["neighbor_list"] = str2class(
            get_alias(calculator_config["neighbor_list"])
        )

    # Build the calculator
    calculator = hydra.utils.instantiate(calculator_config)

    # ===========================================
    #   Set up simulator hooks
    # ===========================================

    simulation_hooks = []

    # Temperature and pressure control should be treated differently (e.g. to avoid double thermostating with
    # NHC barostat and since npt integrators rely on the barostat for system propagation routines)
    if config.dynamics.thermostat is not None:
        log.info("Found {:s} thermostat...".format(config.dynamics.thermostat._target_))
        thermostat_hook = hydra.utils.instantiate(
            config_alias(config.dynamics.thermostat)
        )
    else:
        thermostat_hook = None

    # Check for barostat hook and whether thermostat is required
    if config.dynamics.barostat is not None:
        log.info("Found {:s} barostat...".format(config.dynamics.barostat._target_))
        barostat_hook = hydra.utils.instantiate(config_alias(config.dynamics.barostat))

        if thermostat_hook is not None:
            if hasattr(barostat_hook, "temperature_control"):
                if barostat_hook.temperature_control:
                    log.warning(
                        "Barostat also performs temperature control, ignoring thermostat..."
                    )
                else:
                    simulation_hooks.append(thermostat_hook)
            else:
                log.warning(
                    "Could not determine whether barostat has in-built temperature control. Please"
                    "check for double thermostatting."
                )
                simulation_hooks.append(thermostat_hook)
    else:
        if thermostat_hook is not None:
            simulation_hooks.append(thermostat_hook)
        barostat_hook = None

    # Initialize all other simulation hooks
    simulation_hooks += [
        hydra.utils.instantiate(config_alias(hook_cfg))
        for hook_cfg in config.dynamics.simulation_hooks
    ]

    # ===========================================
    #   Set up integrator
    # ===========================================

    integrator_config = dict(config.dynamics.integrator)
    integrator_config["_target_"] = get_alias(integrator_config["_target_"])

    # Special integrators are needed for NPT simulations
    if barostat_hook is not None:
        integrator_config["_target_"] = get_npt_integrator(
            integrator_config["_target_"]
        )
        integrator_config["barostat"] = barostat_hook

    log.info("Using {:s} integrator...".format(integrator_config["_target_"]))

    # Check for RPMD and set number of replicas
    if is_rpmd_integrator(integrator_config["_target_"]):
        log.info("Setting up ring polymer molecular dynamics...")
        integrator_config["n_replicas"] = system.n_replicas

        # Check if thermostat can be used
        if thermostat_hook is not None:
            if hasattr(thermostat_hook, "ring_polymer"):
                if not thermostat_hook.ring_polymer:
                    raise MDSetupError(
                        "Thermostat not suitable for ring polymer dynamics."
                    )
            else:
                log.warning(
                    "Could not determine if thermostat is suitable for ring polymer dynamics. Good luck."
                )

        # Check barostat
        if barostat_hook is not None:
            if hasattr(barostat_hook, "ring_polymer"):
                if not barostat_hook.ring_polymer:
                    raise MDSetupError(
                        "Barostat not suitable for ring polymer dynamics."
                    )
            else:
                log.warning(
                    "Could not determine if barostat is suitable for ring polymer dynamics. Good luck."
                )

    # Finally build integrator...
    integrator = hydra.utils.instantiate(integrator_config)

    # ===========================================
    #   Set up logging
    # ===========================================

    logging_hooks = []

    for hook in config.logging.logging:
        # Get alias and convert config to dict for instantiating everything
        hook_cfg = dict(config_alias(hook))
        log.info("Setting up {:s} logger...".format(hook_cfg["_target_"]))

        # Initialize data streams for file logger
        if "data_streams" in hook_cfg:
            data_streams = [
                hydra.utils.instantiate(config_alias(data_stream))
                for data_stream in hook_cfg["data_streams"]
            ]
            hook_cfg["data_streams"] = data_streams

        logging_hooks.append(hydra.utils.instantiate(hook_cfg))

    # ===========================================
    #   Set up simulator
    # ===========================================

    simulator = schnetpack.md.Simulator(
        system,
        integrator,
        calculator,
        simulator_hooks=simulation_hooks + logging_hooks,
        gradients_required=False,
    )

    # Set devices and precision
    simulator = simulator.to(device)
    simulator = simulator.to(precision)

    # ===========================================
    #   Finally run simulation
    # ===========================================
    simulator.simulate(config.dynamics.n_steps)

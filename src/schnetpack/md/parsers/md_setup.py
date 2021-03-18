"""
Module for setting up a :obj:`schnetpack.md.simulator.Simulator` from a config dictionary,
which can then be used for simulation.
"""
import logging
import os
import torch
from shutil import rmtree
from ase import io

try:
    import oyaml as yaml
except ImportError:
    import yaml

from schnetpack.md import Simulator, System
from schnetpack.md.parsers.md_options import *
from schnetpack.md.simulation_hooks import *
from schnetpack.md.neighbor_lists import *


class MDSimulation:
    """
    Set up a :obj:`schnetpack.md.Simulator` for performing molecular dynamics simulations
    based on a configuration dictionary. The dictionary is parsed using individual
    :obj:`SetupBlock` instances, which also update the dictionary if default settings are changed.

    Args:
        config (dict): Configuration dictionary either read from an input
                       file in yaml format or generated internally.
    """

    def __init__(self, config):
        self.config = config

        # Setup containers
        self.device = None
        self.simulation_dir = None
        self.integrator = None
        self.system = None
        self.calculator = None
        self.restart = False
        self.load_system_state = False
        self.n_steps = None
        self.hooks = []

        # Initialize device
        SetupDevice(self)

        # Setup seed
        SetupSeed(self)

        # Setup directories
        SetupDirectories(self)

        # Setup the system, including initial conditions
        SetupSystem(self)

        # Get the calculator
        SetupCalculator(self)

        # Get the integrator
        SetupDynamics(self)

        # Get bias potentials if applicable
        SetupBiasPotential(self)

        # Set up adaptive sampling
        SetupAdaptiveSampling(self)

        # Setup Logging
        SetupLogging(self)

        # Get simulator
        self.simulator = self._build_simulator()

    def _build_simulator(self):
        """
        Build the final :obj:`schnetpack.md.Simulator` based on all collected
        subroutines.

        Returns:
            schnetpack.md.Simulator: The simulator for performing the MD.
        """
        simulator = Simulator(self.system, self.integrator, self.calculator, self.hooks)

        # If requested, read restart data
        if self.restart and (self.restart is not None):
            state_dict = torch.load(self.restart, map_location=self.device)
            simulator.restart_simulation(state_dict, soft=False)
            logging.info(f"Restarting simulation from {self.restart}...")
        elif self.load_system_state:
            state_dict = torch.load(self.load_system_state, map_location=self.device)
            simulator.load_system_state(state_dict)
            logging.info(f"Loaded system state from {self.load_system_state}...")

        return simulator

    def run(self):
        """
        Run the simulation using the internal simulator.
        """
        self.simulator.simulate(self.n_steps)

    def save_config(self):
        """
        Save the collected and updated config directory to a yaml file.
        """
        yamlpath = os.path.join(self.simulation_dir, "config.yaml")
        with open(yamlpath, "w") as yf:
            yaml.dump(self.config, yf, default_flow_style=False)


class SetupBlock:
    """
    Base class for defining default entries in the config dictionary as well as
    parsing new instructions. Based on these instructions, the ``_setup`` function is
    used to construct the target simulation module or carry out the requested tasks.
    At the same time, existing config entries are updated.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {}
    target_block = None

    def __init__(self, md_initializer):
        self.target_config_block = self._get_target_block(md_initializer)
        self._load_defaults()
        self._setup(md_initializer)

    def _load_defaults(self):
        """
        Check for default settings in ``self.default_options`` and update config dictionary.
        """
        for option in self.default_options:
            if option not in self.target_config_block:
                self.target_config_block[option] = self.default_options[option]

            elif type(self.default_options[option]) == dict:
                for sub_option in self.default_options[option]:
                    if sub_option not in self.target_config_block[option]:
                        self.target_config_block[option][
                            sub_option
                        ] = self.default_options[option][sub_option]

    def _get_target_block(self, md_initializer):
        """
        Determine the target block the initializer is acting on.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        if self.target_block is not None:
            if self.target_block in md_initializer.config:
                return md_initializer.config[self.target_block]
            else:
                return None
        else:
            return md_initializer.config

    def _setup(self, md_initializer):
        """
        Placeholder function. This is used to set up the individual simulation
        modules and needs to be adapted accordingly.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        raise NotImplementedError


class SetupDirectories(SetupBlock):
    """
    Routine for setting up simulation directories.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {"simulation_dir": "simulation_dir", "overwrite": True}

    def _setup(self, md_initializer):
        """
        Main setup routine, check if directory exists and perform appropriate actions.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        simulation_dir = self.target_config_block["simulation_dir"]
        overwrite = self.target_config_block["overwrite"]

        logging.info("Create model directory")

        if simulation_dir is None:
            raise ValueError("Config `simulation_dir` has to be set!")

        if os.path.exists(simulation_dir) and not overwrite:
            logging.warning(
                "Simulation directory {:s} already exists (set overwrite flag?)".format(
                    simulation_dir
                )
            )

        if os.path.exists(simulation_dir) and overwrite:
            rmtree(simulation_dir)

        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        md_initializer.simulation_dir = simulation_dir


class SetupSeed(SetupBlock):
    """
    Initialize the random seed from the config file.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {"seed": None}

    def _setup(self, md_initializer):
        """
        Initialize the random seed from the config file.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        seed = self.target_config_block["seed"]
        from schnetpack.utils import set_random_seed

        set_random_seed(seed)


class SetupDevice(SetupBlock):
    """
    Determine the computation device.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {"device": "cpu"}

    def _setup(self, md_initializer):
        """
        Determine the computation device from the config.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        device = torch.device(md_initializer.config["device"])
        md_initializer.device = device


class SetupSystem(SetupBlock):
    """
    Parse the system block in the input file and setup the
    atomistic system in :obj:`schnetpack.md.System`.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {
        "n_replicas": 1,
        "molecule_file": "dummy.xyz",
        "initializer": {
            InitialConditionsInit.kind: "maxwell-boltzmann",
            "temperature": 300,
            "remove_translation": True,
            "remove_rotation": True,
        },
        "load_system_state": False,
    }
    target_block = "system"

    def _setup(self, md_initializer):
        """
        Main routine. Loads the molecules and initializes the system.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        molecule_file = self.target_config_block["molecule_file"]
        n_replicas = self.target_config_block["n_replicas"]

        # Read in the molecular structures
        ase_molecules = io.read(molecule_file, index=":")

        # Set up the system
        system = System(n_replicas, device=md_initializer.device)
        system.load_molecules(ase_molecules)

        # Apply initial conditions if requested
        if "initializer" in self.target_config_block:
            initializer = self.target_config_block["initializer"]
            initconds = InitialConditionsInit(initializer)

            if initconds.initialized is not None:
                initconds.initialized.initialize_system(system)

        md_initializer.load_system_state = self.target_config_block["load_system_state"]
        md_initializer.system = system


class SetupCalculator(SetupBlock):
    """
    Parse the calculator block and initialize the calculator for the molecular dynamics simulations.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {
        CalculatorInit.kind: "schnet",
        "required_properties": ["energy", "forces"],
        "force_handle": "forces",
        # "position_conversion": "Angstrom",
        # "force_conversion": "kcal/mol/Angstrom",
        "property_conversion": {},
        # "neighbor_list": "simple",
        # "cutoff": -1.0,
        # "cutoff_shell": 1.0,
    }
    target_block = "calculator"
    basic_models = ["schnet", "custom"]
    ensemble_models = ["schnet_ensemble", "custom_ensemble"]
    neighbor_list = {
        "simple": SimpleNeighborList,
        "ase": ASENeighborList,
        "torch": TorchNeighborList,
    }

    def _setup(self, md_initializer):
        """
        Main routine for loading the model, preparing it for the calculator and setting
        up the main :obj:`schnetpack.md.calculator`.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        calculator = self.target_config_block
        calculator_dict = {}

        # Load model, else get options
        for key in calculator:
            # TODO Figure out conventions for ensemble
            if key == "model_file" or key == "model_files":
                if calculator[CalculatorInit.kind] in self.basic_models:
                    model = self._load_model_schnetpack(
                        calculator["model_file"], md_initializer.device
                    ).to(md_initializer.device)
                    calculator_dict["model"] = model
                elif calculator[CalculatorInit.kind] in self.ensemble_models:
                    models = [
                        self._load_model_schnetpack(model, md_initializer.device).to(
                            md_initializer.device
                        )
                        for model in calculator["model_files"]
                    ]
                    calculator_dict["models"] = models
                elif calculator[CalculatorInit.kind] == "sgdml":
                    model = self._load_model_sgdml(calculator["model_file"]).to(
                        md_initializer.device
                    )
                    calculator_dict["model"] = model
                else:
                    raise ValueError(
                        f"Unrecognized ML calculator {calculator[CalculatorInit.kind]}"
                    )
            elif key == "neighbor_list":
                # Check for neighbor list
                calculator_dict[key] = self.neighbor_list[calculator[key]]
            else:
                calculator_dict[key] = calculator[key]

        calculator = CalculatorInit(calculator_dict).initialized

        md_initializer.calculator = calculator

    @staticmethod
    def _load_model_schnetpack(model_path, device):
        """
        Load the model from a model file and move it to the desired device.

        Args:
            model_path (str): Path to the stored model.
            device (torch.device): Device the computations should be performed on.

        Returns:
            schnetpack.AtomisticModel: SchNetPack model.

        """
        # If model is a directory, search for best_model file
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "best_model")

        # Load model
        model = schnetpack.utils.load_model(model_path, map_location=device)
        logging.info("Loaded model from {:s}".format(model_path))

        return model

    @staticmethod
    def _load_model_sgdml(model_path):
        """
        Load an sGDML model from disk.

        Args:
            model_path (str): Path to the stored sGDML model parameters.

        Returns:
            sgdml.torchtools.GDMLTorchPredict: sGDML model.
        """
        import numpy as np

        try:
            from sgdml.torchtools import GDMLTorchPredict
        except:
            raise ImportError(
                "Could not load sGDML. Please make sure the package is installed."
            )

        try:
            parameters = np.load(model_path, allow_pickle=True)
        except:
            raise ValueError("Could not read sGDML model from {:s}".format(model_path))

        model = GDMLTorchPredict(parameters)
        logging.info("Loaded sGDML model from {:s}".format(model_path))
        return model


class SetupDynamics(SetupBlock):
    """
    Parse the `dynamics` block in the input file and prepare the :obj:`schnetpack.md.Integrator`.
    Also parses thermostats and whether rotational and translational motion should be removed
    during simulation.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {
        "integrator": {IntegratorInit.kind: "verlet", "time_step": 0.5},
        "n_steps": 10000,
        "restart": False,
    }
    target_block = "dynamics"

    def _setup(self, md_initializer):
        """
        Initialize the integrator, as well as all thermostat hooks.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """
        # Get the integrator config
        integrator_config = self.target_config_block["integrator"]

        # Set the device for integrator (used in normal mode transformation)
        integrator_config["device"] = md_initializer.device

        if integrator_config[IntegratorInit.kind] == "ring_polymer":
            integrator_config["n_beads"] = md_initializer.system.n_replicas

        # Add a thermostat if requested (certain baristats have thermostats built in,
        # in this case all options here are ignored)
        if "thermostat" in self.target_config_block:
            thermostat_config = self.target_config_block["thermostat"]
            thermostat = ThermostatInit(thermostat_config).initialized
        else:
            thermostat = None

        if "barostat" in self.target_config_block:
            barostat_config = self.target_config_block["barostat"]

            # Issue message in case a barostat with included thermostat is used
            # -> NHC type barostats
            if barostat_config[BarostatInit.kind] in ["nhc_iso", "nhc_aniso"]:
                logging.info(
                    "The chosen barostat {:s} already includes a thermostat."
                    "All thermostat options will be ignored".format(BarostatInit.kind)
                )
                thermostat = None

            # Automatically set barostat temperature to thermostat temperature if feasible
            if thermostat is not None:
                logging.info(
                    f"Thermostat detected, setting barostat temperature to {thermostat.temperature_bath}K"
                )
                barostat_config["temperature"] = thermostat.temperature_bath

            barostat = BarostatInit(barostat_config).initialized

            # If a barostat of the NHC and BLA kind is used, switch to the corresponding NPT integrator
            if barostat_config is not None:
                if integrator_config[IntegratorInit.kind] == "ring_polymer":
                    integrator_config[IntegratorInit.kind] = "npt_ring_polymer"
                    integrator_config["barostat"] = barostat
                if integrator_config[IntegratorInit.kind] == "verlet":
                    integrator_config[IntegratorInit.kind] = "npt_verlet"
                    integrator_config["barostat"] = barostat
        else:
            barostat = None

        # Finally add barostat and thermostats to hooks
        if barostat is not None:
            md_initializer.hooks.append(barostat)
        if thermostat is not None:
            md_initializer.hooks.append(thermostat)

        # Iniotialize the integrator based on the above modifications
        md_initializer.integrator = IntegratorInit(integrator_config).initialized

        # Remove the motion of the center of motion
        if "remove_com_motion" in self.target_config_block:
            remove_com_config = self.target_config_block["remove_com_motion"]

            if (
                "every_n_steps" not in remove_com_config
                or "remove_rotation" not in remove_com_config
            ):
                raise InitializerError("Missing options in remove_com_motion")
            else:
                md_initializer.hooks += [
                    RemoveCOMMotion(
                        every_n_steps=remove_com_config["every_n_steps"],
                        remove_rotation=remove_com_config["remove_rotation"],
                    )
                ]

        md_initializer.n_steps = self.target_config_block["n_steps"]
        md_initializer.restart = self.target_config_block["restart"]


class SetupBiasPotential(SetupBlock):
    """
    Routine for building a bias potential hook if requested.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {}
    target_block = "bias_potential"

    def _setup(self, md_initializer):
        """
        Check if a bias potential is present and parse the options depending on the type
        of potential. The appropriate hook is then added to those already present.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """

        if "bias_potential" in md_initializer.config:
            bias_potential = self.target_config_block

            # For metadynamics
            if bias_potential[BiasPotentialInit.kind] == "metadyn":
                dummy_potential = {}
                colvars = []
                # Parse the provided collective variable options
                for k, v in bias_potential.items():
                    if k == "colvars":
                        for cv in bias_potential[k]:
                            cv = cv.split()
                            cv_type = cv[0].lower()
                            cv_inputs = [int(x) for x in cv[1:3]]
                            cv_width = float(cv[3])
                            colvars.append(
                                ColVars.available[cv_type](*cv_inputs, cv_width)
                            )
                        dummy_potential["collective_variables"] = colvars
                    else:
                        dummy_potential[k] = v

                md_initializer.hooks += [BiasPotentialInit(dummy_potential).initialized]
            # For accelerated molecular dynamics
            else:
                md_initializer.hooks += [BiasPotentialInit(bias_potential).initialized]


class SetupAdaptiveSampling(SetupBlock):
    default_options = {}
    target_block = "adaptive_sampling"

    def _setup(self, md_initializer):
        if self.target_block in md_initializer.config:
            adaptive_sampling = self.target_config_block

            if adaptive_sampling[AdaptiveSamplingInit.kind] == "ensemble":
                with open(adaptive_sampling["thresholds"], "r") as tf:
                    thresholds = yaml.load(tf)
                adaptive_sampling["thresholds"] = thresholds

            md_initializer.hooks += [
                AdaptiveSamplingInit(adaptive_sampling).initialized
            ]


class SetupLogging(SetupBlock):
    """
    Construct and update the logging hooks.

    Args:
        md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
    """

    default_options = {
        "file_logger": {
            "buffer_size": 100,
            "streams": ["molecules", "properties"],
            "every_n_steps": 1,
        },
        "temperature_logger": 100,
        "write_checkpoints": 1000,
    }
    target_block = "logging"

    def _setup(self, md_initializer):
        """
        Parse the config and create all requested logging hooks.

        Args:
            md_initializer (schnetpack.md.parser.MDSimulation): Parent MDSimulation class.
        """

        if "file_logger" in self.target_config_block:
            logging_file = os.path.join(
                md_initializer.simulation_dir, "simulation.hdf5"
            )
            file_logging_config = self.target_config_block["file_logger"]

            data_streams = get_data_streams(file_logging_config["streams"])

            md_initializer.hooks += [
                FileLogger(
                    logging_file,
                    file_logging_config["buffer_size"],
                    data_streams=data_streams,
                    every_n_steps=file_logging_config["every_n_steps"],
                )
            ]

        tensorboard_logdir = os.path.join(md_initializer.simulation_dir, "logging")

        if "temperature_logger" in self.target_config_block:
            # temperature_dir = os.path.join(md_initializer.simulation_dir, "temperature")
            temperature_dir = tensorboard_logdir
            md_initializer.hooks += [
                TemperatureLogger(
                    temperature_dir,
                    every_n_steps=self.target_config_block["temperature_logger"],
                )
            ]

        if "pressure_logger" in self.target_config_block:
            pressure_dir = tensorboard_logdir
            md_initializer.hooks += [
                PressureLogger(
                    pressure_dir,
                    every_n_steps=self.target_config_block["pressure_logger"],
                )
            ]

        if "write_checkpoints" in self.target_config_block:
            chk_file = os.path.join(md_initializer.simulation_dir, "checkpoint.chk")
            md_initializer.hooks += [
                Checkpoint(
                    chk_file,
                    every_n_steps=self.target_config_block["write_checkpoints"],
                )
            ]

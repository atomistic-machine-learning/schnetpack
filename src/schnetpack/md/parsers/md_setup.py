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
from schnetpack.md.parsers.md_input_parser import *
from schnetpack.md.simulation_hooks import *


class MDInitializer:

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

        # Setup Logging
        SetupLogging(self)

        # Save config
        self.save_config()

    def build_simulator(self):

        simulation = Simulator(self.system, self.integrator, self.calculator, self.hooks)

        # If requested, read restart data
        if self.restart:
            state_dict = torch.load(self.restart)
            simulation.restart(state_dict, soft=False)
            logging.info(f'Restarting simulation from {self.restart}...')
        elif self.load_system_state:
            state_dict = torch.load(self.load_system_state)
            simulation.load_system_state(state_dict)
            logging.info(f'Loaded system state from {self.load_system_state}...')

        return simulation

    def save_config(self):
        yamlpath = os.path.join(self.simulation_dir, 'config.yaml')
        with open(yamlpath, 'w') as yf:
            yaml.dump(self.config, yf, default_flow_style=False)


class SetupBlock:
    default_options = {}
    target_block = None

    def __init__(self, md_initializer):
        self.target_config_block = self._get_target_block(md_initializer)
        self._load_defaults()
        self._setup(md_initializer)

    def _load_defaults(self):
        for option in self.default_options:
            if option not in self.target_config_block:
                self.target_config_block[option] = self.default_options[option]

            elif type(self.default_options[option]) == dict:
                for sub_option in self.default_options[option]:
                    if sub_option not in self.target_config_block[option]:
                        self.target_config_block[option][sub_option] = self.default_options[option][sub_option]

    def _get_target_block(self, md_initializer):
        if self.target_block is not None:
            if self.target_block in md_initializer.config:
                return md_initializer.config[self.target_block]
            else:
                return None
        else:
            return md_initializer.config

    def _setup(self, md_initializer):
        raise NotImplementedError


class SetupDirectories(SetupBlock):
    default_options = {
        'simulation_dir': 'simulation_dir',
        'overwrite': True
    }

    def _setup(self, md_initializer):

        simulation_dir = self.target_config_block['simulation_dir']
        overwrite = self.target_config_block['overwrite']

        logging.info("Create model directory")

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

        md_initializer.simulation_dir = simulation_dir


class SetupSeed(SetupBlock):
    default_options = {
        'seed': None
    }

    def _setup(self, md_initializer):
        seed = self.target_config_block['seed']
        from schnetpack.utils import set_random_seed
        set_random_seed(seed)


class SetupDevice(SetupBlock):
    default_options = {
        'device': 'cpu'
    }

    def _setup(self, md_initializer):
        device = torch.device(md_initializer.config['device'])
        md_initializer.device = device


class SetupSystem(SetupBlock):
    default_options = {
        'n_replicas': 1,
        'molecule_file': 'dummy.xyz',
        'initializer': {
            InitialConditionsInit.kind: 'maxwell-boltzmann',
            'temperature': 300,
            'remove_translation': True,
            'remove_rotation': True,
        }
    }
    target_block = 'system'

    def _setup(self, md_initializer):
        molecule_file = self.target_config_block['molecule_file']
        n_replicas = self.target_config_block['n_replicas']

        # Read in the molecular structures
        ase_molecules = io.read(molecule_file, index=':')

        # Set up the system
        system = System(n_replicas, device=md_initializer.device)
        system.load_molecules(ase_molecules)

        # Apply initial conditions if requested
        if 'initializer' in self.target_config_block:
            initializer = self.target_config_block['initializer']
            initconds = InitialConditionsInit(initializer)

            if initconds.initialized is not None:
                initconds.initialized.initialize_system(system)

        md_initializer.system = system


class SetupCalculator(SetupBlock):
    default_options = {
        CalculatorInit.kind: 'schnet',
        'required_properties': ['energy', 'forces'],
        'force_handle': 'forces',
        'position_conversion': 'Angstrom',
        'force_conversion': 'kcal/mol/Angstrom',
        'property_conversion': {}
    }
    target_block = 'calculator'
    schnet_models = ['schnet']

    def _setup(self, md_initializer):
        calculator = self.target_config_block
        calculator_dict = {}

        # Load model, else get options
        for key in calculator:
            if key == 'model_file':
                if calculator[CalculatorInit.kind] in self.schnet_models:
                    model = self._load_model_schnetpack(calculator['model_file']).to(md_initializer.device)
                elif calculator[CalculatorInit.kind] == 'sgdml':
                    model = self._load_model_sgdml(calculator['model_file']).to(md_initializer.device)
                else:
                    raise ValueError(f'Unrecognized ML calculator {calculator[CalculatorInit.kind]}')
                calculator_dict['model'] = model
            else:
                calculator_dict[key] = calculator[key]

        calculator = CalculatorInit(calculator_dict).initialized

        md_initializer.calculator = calculator

    @staticmethod
    def _load_model_schnetpack(model_path):
        # If model is a directory, search for best_model file
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "best_model")
        logging.info("Loaded model from {:s}".format(model_path))
        model = torch.load(model_path)
        return model

    @staticmethod
    def _load_model_sgdml(model_path):
        import numpy as np

        try:
            from sgdml.torchtools import GDMLTorchPredict
        except:
            raise ImportError('Could not load sGDML. Please make sure the package is installed.')

        try:
            parameters = np.load(model_path)
        except:
            raise ValueError("Could not read sGDML model from {:s}".format(model_path))

        model = GDMLTorchPredict(parameters)
        logging.info('Loaded sGDML model from {:s}'.format(model_path))
        return model


class SetupDynamics(SetupBlock):
    default_options = {
        'integrator': {
            IntegratorInit.kind: 'verlet',
            'time_step': 0.5
        },
        'n_steps': 10000,
        'restart': False,
        'load_system_state': False
    }
    target_block = 'dynamics'

    def _setup(self, md_initializer):
        # Build the integrator
        integrator_config = self.target_config_block['integrator']

        if integrator_config[IntegratorInit.kind] == 'ring_polymer':
            integrator_config['n_beads'] = md_initializer.system.n_replicas

        md_initializer.integrator = IntegratorInit(integrator_config).initialized

        # Add a thermostat if requested
        if 'thermostat' in self.target_config_block:
            thermostat_config = self.target_config_block['thermostat']
            thermostat = ThermostatInit(thermostat_config).initialized
            if thermostat is not None:
                md_initializer.hooks += [thermostat]

        # Remove the motion of the center of motion
        if 'remove_com_motion' in self.target_config_block:
            remove_com_config = self.target_config_block['remove_com_motion']

            if 'every_n_steps' not in remove_com_config or 'remove_rotation' not in remove_com_config:
                raise InitializerError('Missing options in remove_com_motion')
            else:
                md_initializer.hooks += [
                    RemoveCOMMotion(
                        every_n_steps=remove_com_config['every_n_steps'],
                        remove_rotation=remove_com_config['remove_rotation']
                    )
                ]

        md_initializer.n_steps = self.target_config_block['n_steps']
        md_initializer.restart = self.target_config_block['restart']
        md_initializer.load_system_state = self.target_config_block['load_system_state']


class SetupBiasPotential(SetupBlock):
    default_options = {}
    target_block = 'bias_potential'

    def _setup(self, md_initializer):

        if 'bias_potential' in md_initializer.config:
            bias_potential = self.target_config_block

            if bias_potential[BiasPotentialInit.kind] == 'metadyn':
                dummy_potential = {}
                colvars = []
                for k, v in bias_potential.items():
                    if k == 'colvars':
                        for cv in bias_potential[k]:
                            cv = cv.split()
                            cv_type = cv[0].lower()
                            cv_inputs = [int(x) for x in cv[1:3]]
                            cv_width = float(cv[3])
                            colvars.append(ColVars.available[cv_type](*cv_inputs, cv_width))
                        dummy_potential['collective_variables'] = colvars
                    else:
                        dummy_potential[k] = v

                md_initializer.hooks += [BiasPotentialInit(dummy_potential).initialized]
            else:
                md_initializer.hooks += [BiasPotentialInit(bias_potential).initialized]


class SetupLogging(SetupBlock):
    default_options = {
        'file_logger': {
            'buffer_size': 100,
            'streams': ['molecules', 'properties'],
            'every_n_steps': 1
        },
        'temperature_logger': 100,
        'write_checkpoints': 1000
    }
    target_block = 'logging'

    def _setup(self, md_initializer):

        # Convert restart to proper boolean:
        if md_initializer.restart is None or not md_initializer.restart:
            restart = False
        else:
            restart = True

        if 'file_logger' in self.target_config_block:
            logging_file = os.path.join(md_initializer.simulation_dir, 'simulation.hdf5')
            file_logging_config = self.target_config_block['file_logger']

            data_streams = get_data_streams(file_logging_config['streams'])

            md_initializer.hooks += [
                FileLogger(
                    logging_file,
                    file_logging_config['buffer_size'],
                    data_streams=data_streams,
                    restart=restart,
                    every_n_steps=file_logging_config['every_n_steps']
                )
            ]

        if 'temperature_logger' in self.target_config_block:
            temperature_dir = os.path.join(md_initializer.simulation_dir, 'temperature')
            md_initializer.hooks += [
                TemperatureLogger(temperature_dir, every_n_steps=self.target_config_block['temperature_logger'])
            ]

        if 'write_checkpoints' in self.target_config_block:
            chk_file = os.path.join(md_initializer.simulation_dir, 'checkpoint.chk')
            md_initializer.hooks += [
                Checkpoint(chk_file, every_n_steps=self.target_config_block['write_checkpoints'])
            ]

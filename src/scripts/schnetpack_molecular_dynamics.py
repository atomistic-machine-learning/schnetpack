import argparse
import torch
import logging
from shutil import rmtree
import os
from ase import io

from schnetpack.md.parsers.md_input_parser import *
from schnetpack.md import *
from schnetpack.md.simulation_hooks import *

try:
    import oyaml as yaml
except ImportError:
    import yaml


def read_options(yamlpath):
    with open(yamlpath, 'r') as tf:
        tradoffs = yaml.load(tf)

    logging.info('Read options from {:s}.'.format(yamlpath))
    return tradoffs


def save_options(options, yamlpath):
    with open(yamlpath, 'w') as tf:
        yaml.dump(options, tf, default_flow_style=False)


class SetupBlock:
    default_options = {}

    def __init__(self, md_initializer):

        for option in self.default_options:
            if option not in md_initializer.config:
                md_initializer.config[option] = self.default_options[option]

            elif type(option) == dict:
                for sub_option in self.default_options[option]:
                    if sub_option not in md_initializer.config[option]:
                        md_initializer.config[option][sub_option] = self.default_options[option][sub_option]

        self._setup(md_initializer)

    def _setup(self, md_initializer):
        raise NotImplementedError


class SetupDirectories(SetupBlock):
    default_options = {
        'simulation_dir': 'simulation_dir',
        'overwrite': True
    }

    def _setup(self, md_initializer):

        simulation_dir = md_initializer.config['simulation_dir']
        overwrite = md_initializer.config['overwrite']

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
        seed = md_initializer.config['seed']
        from schnetpack.utils import set_random_seed
        set_random_seed(seed)


class SetupDevice(SetupBlock):
    default_options = {
        'device': 'cpu'
    }

    def _setup(self, md_initializer):
        device = torch.device(md_initializer.config['device'])
        md_initializer.device = device


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

        # Store options
        save_options(self.config, os.path.join(self.simulation_dir, 'config.yaml'))

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


class SetupSystem(SetupBlock):
    default_options = {
        'system': {
            'n_replicas': 1,
            'path_to_molecules': 'dummy.xyz'
        },
        'initializer': {
            InitialConditionsInit.kind: 'maxwell-boltzmann',
            'temperature': 300,
            'remove_translation': True,
            'remove_rotation': True,
        }
    }

    def _setup(self, md_initializer):
        molecule_file = md_initializer.config['system']['molecule_file']
        n_replicas = md_initializer.config['system']['n_replicas']

        # Read in the molecular structures
        ase_molecules = io.read(molecule_file, index=':')

        # Set up the system
        system = System(n_replicas, device=md_initializer.device)
        system.load_molecules(ase_molecules)

        # Apply initial conditions if requested
        if 'initializer' in md_initializer.config:
            initializer = md_initializer.config['initializer']
            initconds = InitialConditionsInit(initializer)

            if initconds.initialized is not None:
                initconds.initialized.initialize_system(system)

        md_initializer.system = system


class SetupCalculator(SetupBlock):
    default_options = {
        'calculator': {
            CalculatorInit.kind: 'schnet',
            'required_properties': ['y', 'dydx'],
            'force_handle': 'dydx',
            'position_conversion': 'Angstrom',
            'force_conversion': 'kcal/Angstrom',
            'property_conversion': {},
            'model_path': 'dummy.model'
        }
    }

    def _setup(self, md_initializer):
        calculator = md_initializer.config['calculator']
        calculator_dict = {}
        for key in calculator:
            if key == 'model_file':
                model = torch.load(calculator['model_file']).to(md_initializer.device)
                calculator_dict['model'] = model
            else:
                calculator_dict[key] = calculator[key]

        calculator = CalculatorInit(calculator_dict).initialized

        md_initializer.calculator = calculator


class SetupDynamics(SetupBlock):
    default_options = {
        'simulation': {
            'integrator': {
                IntegratorInit.kind: 'verlet',
                'time_step': 0.5
            },
            'n_steps': 10000,
            'restart': False,
            'load_system_state': False
        }
    }

    def _setup(self, md_initializer):
        # Build the integrator
        integrator = md_initializer.config['simulation']['integrator']

        if integrator[IntegratorInit.kind] == 'ring_polymer':
            integrator['n_beads'] = md_initializer.config['system']['n_replicas']

        md_initializer.integrator = IntegratorInit(integrator).initialized

        # Add a thermostat if requested
        if 'thermostat' in md_initializer.config['simulation']:
            thermostat_config = md_initializer.config['simulation']['thermostat']
            thermostat = ThermostatInit(thermostat_config).initialized
            if thermostat is not None:
                md_initializer.hooks += [thermostat]

        # Remove the motion of the center of motion
        if 'remove_com_motion' in md_initializer.config['simulation']:

            if 'every_n_steps' or 'remove_rotation' not in md_initializer.config['simulation']['remove_com_motion']:
                raise InitializerError('Missing options in remove_com_motion')
            else:
                md_initializer.hooks += [
                    RemoveCOMMotion(
                        every_n_steps=md_initializer.config['simulation']['remove_com_motion']['every_n_steps'],
                        remove_rotation=md_initializer.config['simulation']['remove_com_motion']['remove_rotation']
                    )
                ]

        md_initializer.n_steps = md_initializer.config['simulation']['n_steps']
        md_initializer.restart = md_initializer.config['simulation']['restart']
        md_initializer.load_system_state = md_initializer.config['simulation']['load_system_state']


class SetupBiasPotential(SetupBlock):
    default_options = {}

    def _setup(self, md_initializer):

        if 'bias_potential' in md_initializer.config:
            bias_potential = md_initializer.config['bias_potential']

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
        'logging': {
            'file_logger': {
                'buffer': 200,
                'streams': ['molecules', 'properties'],
                'every_n_steps': 100
            },
            'temperature_logger': 200,
            'write_checkpoints': 1000
        }
    }

    def _setup(self, md_initializer):

        # Convert restart to proper boolean:
        if md_initializer.restart is None or not md_initializer.restart:
            restart = False
        else:
            restart = True

        if 'file_logger' in md_initializer.config['logging']:
            logging_file = os.path.join(md_initializer.simulation_dir, 'simulation.hdf5')

            data_streams = get_data_streams(md_initializer.config['file_logger']['streams'])
            md_initializer.hooks += [
                FileLogger(
                    logging_file,
                    data_streams=data_streams,
                    buffer_size=md_initializer.config['file_logger']['buffer'],
                    restart=restart,
                    every_n_steps=md_initializer.config['file_logger']['every_n_steps']
                )
            ]

        if 'temperature_logger' in md_initializer.config['logging']:
            temperature_dir = os.path.join(md_initializer.simulation_dir, 'temperature')
            md_initializer.hooks += [
                TemperatureLogger(temperature_dir,
                                  every_n_steps=md_initializer.config['logging']['temperature_logger'])
            ]

        if 'write_checkpoints' in md_initializer.config['logging']:
            chk_file = os.path.join(md_initializer.simulation_dir, 'checkpoint.chk')
            md_initializer.hooks += [
                Checkpoint(chk_file, every_n_steps=md_initializer.config['logging']['write_checkpoints'])
            ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('md_input')
    args = parser.parse_args()

    config = read_options(args.md_input)

    mdinit = MDInitializer(config)
    simulation = mdinit.build_simulator()

    # simulation.run(mdinit.n_steps)

    print(config)

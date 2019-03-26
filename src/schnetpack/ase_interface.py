"""
This module provides a ASE calculator class [#ase1]_ for SchNetPack models, as
well as a general Interface to all ASE calculation methods, such as geometry
optimisation, normal mode computation and molecular dynamics simulations.

References
----------
.. [#ase1] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
    Groves, Hammer, Hargus: The atomic simulation environment -- a Python
    library for working with atoms.
    Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import os
from collections import Iterable
from .utils import DeprecationHelper

import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz, write_xyz
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, \
    Stationary, ZeroRotation
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations

from schnetpack.atomistic import Energy, ElementalEnergy, Properties
from schnetpack.data import Structure
from schnetpack.environment import SimpleEnvironmentProvider, \
    collect_atom_triples
from schnetpack.representation import BehlerSFBlock


class MDModelError(Exception):
    pass


class Model:
    """
    Basic wrapper for model to pass the calculator, etc.

    Args:
        model (callable): ML model
        type (str): Model type, allowed is 'schnet'/'wacsf'
        device (str): Device, either GPU or CPU
    """
    implemented = {"wacsf", "schnet"}

    def __init__(self, model, type, device):
        if type not in self.implemented:
            raise NotImplementedError(
                "Unrecognized model type {:s}".format(type))

        self.model = model
        self.type = type
        self.device = device


class SpkCalculator(Calculator):
    """
    ASE calculator for schnetpack machine learning models.

    Args:
        ml_model (object): Model class containing the callable model, device
            and the model type (schnet/wacsf)
        environment_provider (callable): Provides neighbor lists
        pair_provider (callable): Provides list of neighbor pairs. Only
            required if angular descriptors are used. Default is none.
        **kwargs: Additional arguments for basic ase calculator class
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, ml_model,
                 environment_provider=SimpleEnvironmentProvider(), **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model = ml_model.model

        collect_triples = ml_model.type == 'wacsf'
        device = ml_model.device

        self.atoms_converter = \
            AtomsConverter(environment_provider=environment_provider,
                           collect_triples=collect_triples, device=device)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): Properties to calculate.
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        Calculator.calculate(self, atoms)

        # Convert to schnetpack input format
        model_inputs = self.atoms_converter.convert_atoms(atoms)
        # Call model
        model_results = self.model(model_inputs)

        results = {}
        # Convert outputs to calculator format
        if Properties.energy in properties:
            energy = model_results[Properties.energy].cpu().data.numpy()
            results['energy'] = energy.reshape(-1)
        if Properties.forces in properties:
            forces = model_results[Properties.forces].cpu().data.numpy()
            results['forces'] = forces.reshape((len(atoms), 3))

        self.results = results


class AtomsConverter:
    """
    Class to convert ASE atoms object to an input suitable for the SchNetPack
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        pair_provider (callable): Neighbor pair provider (required for angular
            functions)
        device (str): Device for computation (default='cpu')
    """

    def __init__(self, environment_provider=SimpleEnvironmentProvider(),
                 collect_triples=False, device=torch.device('cpu')):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples

        # Get device
        self.device = device

    def convert_atoms(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = {}

        # Elemental composition
        inputs[Structure.Z] = torch.LongTensor(atoms.numbers.astype(np.int))
        inputs[Structure.atom_mask] = \
            torch.ones_like(inputs[Structure.Z]).float()

        # Set positions
        positions = atoms.positions.astype(np.float32)
        inputs[Structure.R] = torch.FloatTensor(positions)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(atoms)

        # Get neighbors and neighbor mask
        mask = torch.FloatTensor(nbh_idx) >= 0
        inputs[Structure.neighbor_mask] = mask.float()
        inputs[Structure.neighbors] = \
            torch.LongTensor(nbh_idx.astype(np.int)) * mask.long()

        # Get cells
        inputs[Structure.cell] = \
            torch.FloatTensor(atoms.cell.astype(np.float32))
        inputs[Structure.cell_offset] = \
            torch.FloatTensor(offsets.astype(np.float32))

        # Set index
        # inputs['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

        # If requested get masks and neighbor lists for neighbor pairs
        if self.collect_triples is not None:
            nbh_idx_j, nbh_idx_k = collect_atom_triples(nbh_idx)
            inputs[Structure.neighbor_pairs_j] = \
                torch.LongTensor(nbh_idx_j.astype(np.int))
            inputs[Structure.neighbor_pairs_k] = \
                torch.LongTensor(nbh_idx_k.astype(np.int))
            inputs[Structure.neighbor_pairs_mask] = \
                torch.ones_like(inputs[Structure.neighbor_pairs_j]).float()

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.unsqueeze(0).to(self.device)

        return inputs


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)

    Args:
        molecule_path (str): Path to initial geometry
        ml_model (object): Model class wrapper for the ML model, type and the
            device
        working_dir (str): Path to directory where files should be stored
    """

    def __init__(self, molecule_path, ml_model, working_dir):
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = None
        self._load_molecule(molecule_path)

        # Set up calculator
        calculator = SpkCalculator(ml_model)
        self.molecule.set_calculator(calculator)

        # Unless initialized, set dynamics to False
        self.dynamics = False

    def _load_molecule(self, molecule_path):
        """
        Load molecule from file (can handle all ase formats).

        Args:
            molecule_path (str): Path to molecular geometry
        """
        file_format = os.path.splitext(molecule_path)[-1]
        if file_format == 'xyz':
            self.molecule = read_xyz(molecule_path)
        else:
            self.molecule = read(molecule_path)

    def save_molecule(self, name, file_format='xyz', append=False):
        """
        Save the current molecular geometry.

        Args:
            name (str): Name of save-file.
            file_format (str): Format to store geometry (default xyz).
            append (bool): If set to true, geometry is added to end of file
                (default False).
        """
        molecule_path = os.path.join(self.working_dir,
                                     "%s.%s" % (name, file_format))
        if file_format == "xyz":
            # For extended xyz format, plain is needed since ase can not parse
            # the extxyz it writes
            write_xyz(molecule_path, self.molecule, plain=True)
        else:
            write(molecule_path, self.molecule, format=file_format,
                  append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule('single_point', file_format='extxyz')

    def init_md(self, name, time_step=0.5, temp_init=300, temp_bath=None,
                reset=False, interval=1):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.

        Args:
            name (str): Basic name of logfile and trajectory
            time_step (float): Time step in fs (default=0.5)
            temp_init (float): Initial temperature of the system in K
                (default is 300)
            temp_bath (float): Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset (bool): Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval (int): Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitely requested via restart=True
        if not self.dynamics or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(self.molecule, time_step * units.fs,
                                     temp_bath * units.kB, 0.01)

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "%s.log" % name)
        trajfile = os.path.join(self.working_dir, "%s.traj" % name)
        logger = MDLogger(self.dynamics, self.molecule, logfile, stress=False,
                          peratom=False, header=True, mode='a')
        trajectory = Trajectory(trajfile, 'w', self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(self, temp_init=300, remove_translation=True,
                         remove_rotation=True):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init (float): Initial temperature in Kelvin (default 300)
            remove_translation (bool): Remove translation components of
                velocity (default True)
            remove_rotation (bool): Remove rotation components of velocity
                (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.

        Args:
            steps (int): Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError("Dynamics need to be initialized using the"
                                 " 'setup_md' function")

        self.dynamics.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
            fmax (float): Maximum residual force change (default 1.e-2)
            steps (int): Maximum number of steps (default 1000)
        """
        name = 'optimization'
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = \
            QuasiNewton(self.molecule, trajectory='%s.traj' % optimize_file,
                        restart='%s.pkl' % optimize_file)
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name)

    def compute_normal_modes(self, write_jmol=True):
        """
        Use ase calculator to compute numerical frequencies for the molecule

        Args:
            write_jmol (bool): Write frequencies to input file for
                visualization in jmol (default=True)
        """
        freq_file = os.path.join(self.working_dir, "normal_modes")

        # Compute frequencies
        frequencies = Vibrations(self.molecule, name=freq_file)
        frequencies.run()

        # Print a summary
        frequencies.summary()

        # Write jmol file if requested
        if write_jmol:
            frequencies.write_jmol()


def load_model(modelpath, cuda=True):
    """
    Load an exported model and prepare it for simulations with ASE. The model
    needs to be able to provide energies and forces.

    Args:
        modelpath (str): Path to exported model files.
        cuda (bool): Use cuda (default=True).

    Returns:
        object: Model class specified in molecular_dynamics. Contains the model,
            model type and device.
    """

    # Set cuda if requested
    device = torch.device("cuda" if cuda else "cpu")
    model = torch.load(modelpath).to(device)

    # Determine model type
    if isinstance(model.representation, BehlerSFBlock):
        model_type = 'wacsf'
    else:
        model_type = 'schnet'

    # Set gradiant flags properly
    model.requires_dr = True

    has_energy = False
    if isinstance(model.output_modules, Iterable):
        for module in model.output_modules:
            if isinstance(module, Energy) or isinstance(module,
                                                        ElementalEnergy):
                has_energy = True
            module.requires_dr = True
    else:
        if isinstance(model.output_modules, Energy) or \
                isinstance(model.output_modules, ElementalEnergy):
            has_energy = True
        model.output_modules.requires_dr = True

    if not has_energy:
        raise MDModelError(
            'Molecular dynamics model requires an Energy/ElementalEnergy '
            'output layer for predicting forces.')

    # Store into model wrapper for calculator
    ml_model = Model(model, model_type, device)

    return ml_model

MLPotential = DeprecationHelper(SpkCalculator, 'MLPotential')

import torch
import os
import torch.nn as nn
import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz, write_xyz
from ase.calculators.calculator import Calculator
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from schnetpack.data import Structure
from schnetpack.atomistic import AtomisticModel, Energy, DipoleMoment
from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples
from schnetpack.config_model import Hyperparameters



class NNCalculator(Calculator, Hyperparameters):
    """
    ASE calculator for schnetpack machine learning models.

    Args:
        representation (nn.Module): Model class containing the callable model, device and the model type (schnet/wacsf)
        n_in (int): input dimension of representation
        calc_energy (bool): calculate energy if True
        calc_forces (bool): calculate forces if True
        calc_dipole (bool): calculate dipole momentum if True
        calc_charges (bool): calculate charges if True
        **kwargs: Additional arguments for basic ase calculator class
    """

    implemented_properties = ['energy', 'forces', 'charges', 'dipole']
    default_parameters = {}

    def __init__(self, representation, model_path=None, n_in=128, calc_energy=False, calc_forces=False,
                 calc_dipole=False, calc_charges=False, load_model_params=False):
        Hyperparameters.__init__(self, locals())
        Calculator.__init__(self)
        self.calc_energy = calc_energy
        self.calc_forces = calc_forces
        self.calc_dipole = calc_dipole
        self.calc_charges = calc_charges

        self.calc_properties = dict(energy=self.calc_energy, forces=self.calc_forces, charges=self.calc_charges,
                                    dipole=self.calc_dipole)

        output_modules = nn.ModuleList()
        if self.calc_dipole or self.calc_charges:
            output_modules.append(DipoleMoment(n_in=n_in, return_charges=calc_charges))
        if self.calc_energy or self.calc_forces:
            output_modules.append(Energy(n_in=n_in, return_force=calc_forces))

        self.atoms_converter = AtomsConverter()

        self.model_path = model_path
        self.model = AtomisticModel(representation=representation, output_modules=output_modules)
        if load_model_params:
            self.load_model()


    def calculate(self, atoms=None, properties=None,
                  system_changes=None):
        """
        Recalculate properties.

        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): Not implemented.
            system_changes (list of str): Not implemented.
        """
        super(NNCalculator, self).calculate(atoms=atoms)

        inputs = self.atoms_converter.convert_atoms(atoms)
        results = self.model(inputs)
        self.results = dict()

        for prop in self.implemented_properties:
            if self.calc_properties[prop]:
                self.results[prop] = results[prop].cpu().detach().numpy().squeeze()

    def get_model(self):
        return self.model

    def load_model(self):
        assert self.model_path is not None, 'Model Path is not defined!'
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, 'best_model')))


class AtomsConverter:
    """
    Class to convert ASE atoms object to an input suitable for the SchNetPack ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        pair_provider (callable): Neighbor pair provider (required for angular functions)
        device (str): Device for computation (default='cpu')
    """

    def __init__(self, environment_provider=SimpleEnvironmentProvider(), collect_triples=False,
                 device=torch.device('cpu')):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples

        # Get device
        self.device = device

    def convert_atoms(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks reformated into SchNetPack
                input format.
        """
        inputs = {}
        idx = 0

        # Elemental composition
        inputs[Structure.Z] = torch.LongTensor(atoms.numbers.astype(np.int))
        inputs[Structure.atom_mask] = torch.ones_like(inputs[Structure.Z]).float()

        # Set positions
        positions = atoms.positions.astype(np.float32)
        inputs[Structure.R] = torch.FloatTensor(positions)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(idx, atoms)

        # Get neighbors and neighbor mask
        mask = torch.FloatTensor(nbh_idx) >= 0
        inputs[Structure.neighbor_mask] = mask.float()
        inputs[Structure.neighbors] = torch.LongTensor(nbh_idx.astype(np.int)) * mask.long()

        # Get cells
        inputs[Structure.cell] = torch.FloatTensor(atoms.cell.astype(np.float32))
        inputs[Structure.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))

        # Set index
        inputs['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

        # If requested get masks and neighbor lists for neighbor pairs
        if self.collect_triples is not None:
            nbh_idx_j, nbh_idx_k = collect_atom_triples(nbh_idx)
            inputs[Structure.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            inputs[Structure.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))
            inputs[Structure.neighbor_pairs_mask] = torch.ones_like(inputs[Structure.neighbor_pairs_j]).float()

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.unsqueeze(0).to(self.device)

        return inputs


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)

    Args:
        molecule_path (str): Path to initial geometry
        representation (nn.Modules): Network representation
        working_dir (str): Path to directory where files should be stored
    """

    def __init__(self, molecule_path, representation, working_dir, calc_energy=True, calc_forces=True,
                 calc_charges=False, calc_dipole=False):
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = None
        self._load_molecule(molecule_path)

        # Set up calculator
        calculator = NNCalculator(representation, calc_energy=calc_energy, calc_forces=calc_forces,
                                  calc_dipole=calc_dipole, calc_charges=calc_charges)
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
            append (bool): If set to true, geometry is added to end of file (default False).
        """
        molecule_path = os.path.join(self.working_dir, "%s.%s" % (name, file_format))
        if file_format == "xyz":
            # For extended xyz format, plain is needed since ase can not parse the extxyz it writes
            write_xyz(molecule_path, self.molecule, plain=True)
        else:
            write(molecule_path, self.molecule, format=file_format, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and store them to the working directory.
        The format used is the extended xyz format. This functionality is mainly intended to be used for interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule('single_point', file_format='extxyz')

    def init_md(self, name, time_step=0.5, temp_init=300, temp_bath=None, reset=False, interval=1):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to be specified, so that old trajectories
        are not overwritten. This functionality can be used to subsequently carry out equilibration and production.

        Args:
            name (str): Basic name of logfile and trajectory
            time_step (float): Time step in fs (default=0.5)
            temp_init (float): Initial temperature of the system in K (default is 300)
            temp_bath (float): Carry out Langevin NVT dynamics at the specified temperature. If set to None, NVE
                               dynamics are performed instead (default=None)
            reset (bool): Whether dynamics should be restarted with new initial conditions (default=False)
            interval (int): Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize velocities unless explicitely requested
        # via restart=True
        if not self.dynamics or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(self.molecule, time_step * units.fs, temp_bath * units.kB, 0.01)

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "%s.log" % name)
        trajfile = os.path.join(self.working_dir, "%s.traj" % name)
        logger = MDLogger(self.dynamics, self.molecule, logfile, stress=False, peratom=False, header=True, mode='a')
        trajectory = Trajectory(trajfile, 'w', self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(self, temp_init=300, remove_translation=True, remove_rotation=True):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init (float): Initial temperature in Kelvin (default 300)
            remove_translation (bool): Remove translation components of velocity (default True)
            remove_rotation (bool): Remove rotation components of velocity (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps):
        """
        Perform a molecular dynamics simulation using the settings specified upon initializing the class.

        Args:
            steps (int): Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError("Dynamics need to be initialized using the 'setup_md' function")

        self.dynamics.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase (BFGS + line search)

        Args:
            fmax (float): Maximum residual force change (default 1.e-2)
            steps (int): Maximum number of steps (default 1000)
        """
        name = 'optimization'
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(self.molecule, trajectory='%s.traj' % optimize_file, restart='%s.pkl' % optimize_file)
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name)

    def compute_normal_modes(self, write_jmol=True):
        """
        Use ase calculator to compute numerical frequencies for the molecule

        Args:
            write_jmol (bool): Write frequencies to input file for visualization in jmol (default=True)
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


'''
def load_model(modelpath, cuda=True):
    """
    Load a trained model from its directory and prepare it for simulations with ASE.

    Args:
        modelpath (str): Path to model directory.
        cuda (bool): Use cuda (default=True).

    Returns:
        object: Model class specified in molecular_dynamics. Contains the model, model type and device.
    """
    # Load stored arguments
    argspath = os.path.join(modelpath, 'args.json')
    args = read_from_json(argspath)

    # Reconstruct model based on arguments
    if args.model == 'schnet':
        representation = SchNet(args.features, args.features, args.interactions,
                                args.cutoff, args.num_gaussians)
        atomwise_output = Energy(args.features, return_force=True, create_graph=True)
    elif args.model == 'wacsf':
        # Build HDNN model
        mode = ('weighted', 'Behler')[args.behler]
        # Convert element strings to atomic charges
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        representation = BehlerSFBlock(args.radial, args.angular, zetas=set(args.zetas), cutoff_radius=args.cutoff,
                                       centered=args.centered, crossterms=args.crossterms, mode=mode,
                                       elements=elements)
        representation = StandardizeSF(representation, cuda=args.cuda)
        atomwise_output = ElementalEnergy(representation.n_symfuncs, n_hidden=args.n_nodes, n_layers=args.n_layers,
                                          return_force=True, create_graph=True, elements=elements)
    else:
        raise ValueError('Unknown model class:', args.model)

    model = AtomisticModel(representation, atomwise_output)

    # Load old parameters
    model.load_state_dict(torch.load(os.path.join(modelpath, 'best_model')))

    # Set cuda if requested
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)

    # Store into model wrapper for calculator
    ml_model = Model(model, args.model, device)

    return ml_model
'''

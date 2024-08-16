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

import ase
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations

import torch
import schnetpack
import logging
from copy import deepcopy

import schnetpack.task
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CastTo32, CastTo64
from schnetpack.units import convert_units
from schnetpack.utils import load_model
from schnetpack.md.utils import activate_model_stress

from typing import Optional, List, Union, Dict
from ase import Atoms

log = logging.getLogger(__name__)

__all__ = ["SpkCalculator", "AseInterface", "AtomsConverter"]


class AtomsConverterError(Exception):
    pass


class AtomsConverter:
    """
    Convert ASE atoms to SchNetPack input batch format for model prediction.

    """

    def __init__(
        self,
        neighbor_list: Union[schnetpack.transform.Transform, None],
        transforms: Union[
            schnetpack.transform.Transform, List[schnetpack.transform.Transform]
        ] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            neighbor_list (schnetpack.transform.Transform, None): neighbor list transform. Can be set to None incase
                that the neighbor list is contained in transforms.
            transforms: transforms for manipulating the neighbor lists. This can be either a single transform or a list
                of transforms that will be executed after the neighbor list is calculated. Such transforms may be
                useful, e.g., for filtering out certain neighbors. In case transforms are required before the neighbor
                list is calculated, neighbor_list argument can be set to None and a list of transforms including the
                neighbor list can be passed as transform argument. The transforms will be executed in the order of
                their appearance in the list.
            device (str, torch.device): device on which the model operates (default: cpu).
            dtype (torch.dtype): required data type for the model input (default: torch.float32).
            additional_inputs (dict): additional inputs required for some transforms.
                When setting up the AtomsConverter, those additional inputs will be
                stored to the input batch.
        """

        self.neighbor_list = deepcopy(neighbor_list)
        self.device = device
        self.dtype = dtype
        self.additional_inputs = additional_inputs or {}

        # convert transforms and neighbor_list to list
        transforms = transforms or []
        if type(transforms) != list:
            transforms = [transforms]
        neighbor_list = [] if neighbor_list is None else [neighbor_list]

        # get transforms and initialize neighbor list
        self.transforms: List[schnetpack.transform.Transform] = (
            neighbor_list + transforms
        )

        # Set numerical precision
        if dtype == torch.float32:
            self.transforms.append(CastTo32())
        elif dtype == torch.float64:
            self.transforms.append(CastTo64())
        else:
            raise AtomsConverterError(f"Unrecognized precision {dtype}")

    def __call__(self, atoms: List[Atoms] or Atoms):
        """

        Args:
            atoms (list or ase.Atoms): list of ASE atoms objects or single ASE atoms object.

        Returns:
            dict[str, torch.Tensor]: input batch for model.
        """

        # check input type and prepare for conversion
        if type(atoms) == list:
            pass
        elif type(atoms) == ase.Atoms:
            atoms = [atoms]
        else:
            raise TypeError(
                "atoms is type {}, but should be either list or ase.Atoms object".format(
                    type(atoms)
                )
            )

        inputs_batch = []
        for at_idx, at in enumerate(atoms):

            inputs = {
                properties.n_atoms: torch.tensor([at.get_global_number_of_atoms()]),
                properties.Z: torch.from_numpy(at.get_atomic_numbers()),
                properties.R: torch.from_numpy(at.get_positions()),
                properties.cell: torch.from_numpy(at.get_cell().array).view(-1, 3, 3),
                properties.pbc: torch.from_numpy(at.get_pbc()).view(-1, 3),
            }

            # specify sample index
            inputs.update({properties.idx: torch.tensor([at_idx])})

            # add additional inputs (specified in AtomsConverter __init__)
            inputs.update(self.additional_inputs)

            for transform in self.transforms:
                inputs = transform(inputs)
            inputs_batch.append(inputs)

        inputs = _atoms_collate_fn(inputs_batch)

        # Move input batch to device
        inputs = {p: inputs[p].to(self.device) for p in inputs}

        return inputs


class SpkCalculatorError(Exception):
    pass


class SpkCalculator(Calculator):
    """
    ASE calculator for schnetpack machine learning models.

    """

    energy = "energy"
    forces = "forces"
    stress = "stress"
    implemented_properties = [energy, forces, stress]

    def __init__(
        self,
        model_file: str,
        neighbor_list: schnetpack.transform.Transform,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: callable = AtomsConverter,
        transforms: Union[
            schnetpack.transform.Transform, List[schnetpack.transform.Transform]
        ] = None,
        additional_inputs: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            model_file (str): path to trained model
            neighbor_list (schnetpack.transform.Transform): SchNetPack neighbor list
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            stress_key (str): name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit (str, float): energy units used by model (default="kcal/mol")
            position_unit (str, float): position units used by model (default="Angstrom")
            device (torch.device): device used for calculations (default="cpu")
            dtype (torch.dtype): select model precision (default=float32)
            converter (callable): converter used to set up input batches
            transforms (schnetpack.transform.Transform, list): transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            additional_inputs (dict): additional inputs required for some transforms in the converter.
            **kwargs: Additional arguments for basic ase calculator class
        """
        Calculator.__init__(self, **kwargs)

        self.converter = converter(
            neighbor_list=neighbor_list,
            device=device,
            dtype=dtype,
            transforms=transforms,
            additional_inputs=additional_inputs,
        )

        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # Mapping between ASE names and model outputs
        self.property_map = {
            self.energy: energy_key,
            self.forces: force_key,
            self.stress: stress_key,
        }

        self.model = self._load_model(model_file)
        self.model.to(device=device, dtype=dtype)

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy: self.energy_conversion,
            self.forces: self.energy_conversion / self.position_conversion,
            self.stress: self.energy_conversion / self.position_conversion**3,
        }

        # Container for basic ml model ouputs
        self.model_results = None

    def _load_model(self, model_file: str) -> schnetpack.model.AtomisticModel:
        """
        Load an individual model, activate stress computation

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        """

        log.info("Loading model from {:s}".format(model_file))
        # load model and keep it on CPU, device can be changed afterwards
        model = load_model(model_file, device=torch.device("cpu")).to(torch.float64)
        model = model.eval()

        if self.stress_key is not None:
            log.info("Activating stress computation...")
            model = activate_model_stress(model, self.stress_key)

        return model

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): select properties computed and stored to results.
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)

            # Convert to schnetpack input format
            model_inputs = self.converter(atoms)
            model_results = self.model(model_inputs)

            results = {}
            # TODO: use index information to slice everything properly
            for prop in properties:
                model_prop = self.property_map[prop]

                if model_prop in model_results:
                    if prop == self.energy:
                        # ase calculator should return scalar energy
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy().item()
                            * self.property_units[prop]
                        )
                    elif prop == self.stress:
                        # squeeze stress dimension [1, 3, 3] of spk to [3, 3] of ase
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy().squeeze()
                            * self.property_units[prop]
                        )
                    else:
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy()
                            * self.property_units[prop]
                        )
                else:
                    raise AtomsConverterError(
                        "'{:s}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(prop)
                    )

            self.results = results
            self.model_results = model_results


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    """

    def __init__(
        self,
        molecule_path: str,
        working_dir: str,
        model_file: str,
        neighbor_list: schnetpack.transform.Transform,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: AtomsConverter = AtomsConverter,
        optimizer_class: type = QuasiNewton,
        fixed_atoms: Optional[List[int]] = None,
        transforms: Union[
            schnetpack.transform.Transform, List[schnetpack.transform.Transform]
        ] = None,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            molecule_path: Path to initial geometry
            working_dir: Path to directory where files should be stored
            model_file (str): path to trained model
            neighbor_list (schnetpack.transform.Transform): SchNetPack neighbor list
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            stress_key (str): name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit (str, float): energy units used by model (default="kcal/mol")
            position_unit (str, float): position units used by model (default="Angstrom")
            device (torch.device): device used for calculations (default="cpu")
            dtype (torch.dtype): select model precision (default=float32)
            converter (schnetpack.interfaces.AtomsConverter): converter used to set up input batches
            optimizer_class (ase.optimize.optimizer): ASE optimizer used for structure relaxation.
            fixed_atoms (list(int)): list of indices corresponding to atoms with positions fixed in space.
            transforms (schnetpack.transform.Transform, list): transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            additional_inputs (dict): additional inputs required for some transforms in the converter.
        """
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = read(molecule_path)

        # Apply position constraints
        if fixed_atoms:
            c = FixAtoms(fixed_atoms)
            self.molecule.set_constraint(constraint=c)

        # Set up optimizer
        self.optimizer_class = optimizer_class

        # Set up calculator
        calculator = SpkCalculator(
            model_file=model_file,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            device=device,
            dtype=dtype,
            converter=converter,
            transforms=transforms,
            additional_inputs=additional_inputs,
        )

        self.molecule.set_calculator(calculator)

        self.dynamics = None

    def save_molecule(self, name: str, file_format: str = "xyz", append: bool = False):
        """
        Save the current molecular geometry.

        Args:
            name: Name of save-file.
            file_format: Format to store geometry (default xyz).
            append: If set to true, geometry is added to end of file (default False).
        """
        molecule_path = os.path.join(
            self.working_dir, "{:s}.{:s}".format(name, file_format)
        )
        write(molecule_path, self.molecule, format=file_format, append=append)

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

        self.save_molecule("single_point", file_format="xyz")

    def init_md(
        self,
        name: str,
        time_step: float = 0.5,
        temp_init: float = 300,
        temp_bath: Optional[float] = None,
        reset: bool = False,
        interval: int = 1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.

        Args:
            name: Basic name of logfile and trajectory
            time_step: Time step in fs (default=0.5)
            temp_init: Initial temperature of the system in K (default is 300)
            temp_bath: Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset: Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval: Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if self.dynamics is None or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(
                self.molecule,
                time_step * units.fs,
                temp_bath * units.kB,
                1.0 / (100.0 * units.fs),
            )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "{:s}.log".format(name))
        trajfile = os.path.join(self.working_dir, "{:s}.traj".format(name))
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(
        self,
        temp_init: float = 300,
        remove_translation: bool = True,
        remove_rotation: bool = True,
    ):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init: Initial temperature in Kelvin (default 300)
            remove_translation: Remove translation components of velocity (default True)
            remove_rotation: Remove rotation components of velocity (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps: int):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.

        Args:
            steps: Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError(
                "Dynamics need to be initialized using the" " 'setup_md' function"
            )

        self.dynamics.run(steps)

    def optimize(self, fmax: float = 1.0e-2, steps: int = 1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
            fmax: Maximum residual force change (default 1.e-2)
            steps: Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = self.optimizer_class(
            self.molecule,
            trajectory="{:s}.traj".format(optimize_file),
            restart="{:s}.pkl".format(optimize_file),
        )
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name, file_format="extxyz")

    def compute_normal_modes(self, write_jmol: bool = True):
        """
        Use ase calculator to compute numerical frequencies for the molecule

        Args:
            write_jmol: Write frequencies to input file for visualization in jmol (default=True)
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

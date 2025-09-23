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
import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import logging
from copy import deepcopy

from ase import Atoms
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

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CastTo32, CastTo64, Transform
from schnetpack.units import convert_units
from schnetpack.utils import load_model
from schnetpack.md.utils import activate_model_stress

from typing import Optional, List, Union, Dict

log = logging.getLogger(__name__)

__all__ = ["SpkCalculator", "AseInterface", "AtomsConverter", "SpkEnsembleCalculator"]


class AtomsConverterError(Exception):
    pass


class AtomsConverter:
    """
    Convert ASE atoms to SchNetPack input batch format for model prediction.

    """

    def __init__(
        self,
        neighbor_list: Union[Transform, None],
        transforms: Union[Transform, List[Transform]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            neighbor_list (Transform, None): neighbor list transform. Can be set to None incase
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
        self.transforms: List[Transform] = neighbor_list + transforms

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
        elif type(atoms) == Atoms:
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
        model: Union[str, nn.Module],
        neighbor_list: Transform,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: callable = AtomsConverter,
        transforms: Union[Transform, List[Transform]] = None,
        additional_inputs: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            model: either path to trained model or model object
            neighbor_list: SchNetPack neighbor list
            energy_key: name of energies in model (default="energy")
            force_key: name of forces in model (default="forces")
            stress_key: name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit: energy units used by model (default="kcal/mol")
            position_unit: position units used by model (default="Angstrom")
            device: device used for calculations (default="cpu")
            dtype: select model precision (default=float32)
            converter: converter used to set up input batches
            transforms: transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            additional_inputs: additional inputs required for some transforms in the converter.
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

        self.model = self._load_model(model, device, dtype)

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

    def _load_model(
        self,
        model: Union[str, nn.Module],
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> nn.Module:
        """
        Load an individual model, activate stress computation

        Args:
            model: Either path to model or model object

        Returns:
           AtomisticTask: loaded schnetpack model
        """

        if isinstance(model, str):
            log.info("Loading model from {:s}...".format(model))
            model = load_model(model, device=torch.device(device)).to(dtype)

        else:
            log.info("Using instantiated model...")
            model = model.to(device).to(dtype)

        model = model.eval()

        if self.stress_key is not None:
            log.info("Activating stress computation...")
            model = activate_model_stress(model, self.stress_key)

        return model

    def calculate(
        self,
        atoms: Atoms = None,
        # properties is just a placeholder and will be ignored
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        """
        Args:
            atoms: ASE atoms object.
            properties: Ignored. Instead, all specified property keys (energy_key, force_key, ...)
                are calculated and stored.
            system_changes: List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        # make a list of all properties available in the model
        properties = [
            p_key for p_key, p_value in self.property_map.items() if p_value is not None
        ]

        # check if calculation is needed
        if not self.calculation_required(atoms, properties):
            return

        # prepare calculation
        Calculator.calculate(self, atoms)
        model_inputs = self.converter(atoms)

        # calculate properties
        model_results = self.model(model_inputs)
        results = {}
        # TODO: use index information to slice everything properly

        # reshape to match ase format and do unit conversion
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


class Uncertainty(ABC):
    def __init__(
        self,
        energy_key="energy",
        force_key="forces",
        stress_key="stress",
        energy_weight=0.0,
        force_weight=1.0,
        stress_weight=0.0,
    ):
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # normalize weights
        total_weight = energy_weight + force_weight + stress_weight
        if total_weight == 0:
            raise ValueError("total_weight cannot be zero")

        self.energy_weight = energy_weight / total_weight
        self.force_weight = force_weight / total_weight
        self.stress_weight = stress_weight / total_weight

    @abstractmethod
    def __call__(self, predictions: Dict[str, List[np.ndarray]]) -> float:
        pass


class AbsoluteUncertainty(Uncertainty):

    def __call__(self, predictions: Dict[str, List[np.ndarray]]) -> float:
        uncertainty = 0

        if self.energy_weight > 0:
            energy_unc = np.std(predictions[self.energy_key])
            uncertainty += self.energy_weight * energy_unc

        if self.force_weight > 0:
            # get per atom uncertainty with L2 norm of stds
            force_std = np.std(predictions[self.force_key], axis=0)
            per_atom_uncertainty = np.linalg.norm(force_std, axis=1)
            # aggregate to scalar uncertainty
            force_unc = np.mean(per_atom_uncertainty)
            uncertainty += self.force_weight * force_unc

        if self.stress_weight > 0:
            # get uncertainty per plane
            stress_std = np.std(predictions[self.stress_key], axis=0)
            per_plane_uncertainty = np.linalg.norm(stress_std, axis=1)
            # aggregate to scalar uncertainty
            stress_unc = np.mean(per_plane_uncertainty)
            uncertainty += self.stress_weight * stress_unc

        return uncertainty


class RelativeUncertainty(Uncertainty):

    def __call__(self, predictions: Dict[str, List[np.ndarray]]) -> float:
        uncertainty = 0

        if self.energy_weight > 0:
            energy_preds = predictions[self.energy_key]
            mean_energy = np.mean(energy_preds)
            std_energy = np.std(energy_preds)
            energy_unc = std_energy / (abs(mean_energy) + 1e-8)
            uncertainty += self.energy_weight * energy_unc

        if self.force_weight > 0:
            force_preds = np.array(predictions[self.force_key])
            mean_force = np.mean(force_preds, axis=0)
            std_force = np.std(force_preds, axis=0)

            mean_norms = np.linalg.norm(mean_force, axis=1)
            std_norms = np.linalg.norm(std_force, axis=1)

            # aggregate to scalar
            force_unc = np.mean(std_norms / (mean_norms + 1e-8))
            uncertainty += self.force_weight * force_unc

        if self.stress_weight > 0:
            stress_preds = np.array(predictions[self.stress_key])
            mean_stress = np.mean(stress_preds, axis=0)
            std_stress = np.std(stress_preds, axis=0)

            mean_planes = np.linalg.norm(mean_stress, axis=1)
            std_planes = np.linalg.norm(std_stress, axis=1)

            # aggregate to scalar
            stress_unc = np.mean(std_planes / (mean_planes + 1e-8))
            uncertainty += self.stress_weight * stress_unc

        return uncertainty


class SpkEnsembleCalculator(SpkCalculator):
    """
    Calculator for neural network models for ensemble calculations.
    Requires multiple models
    """

    def __init__(
        self,
        models: Union[List[str], List[nn.Module]],
        neighbor_list: Transform,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: callable = AtomsConverter,
        transforms: Optional[Union[Transform, List[Transform]]] = None,
        uncertainty_fn: callable = None,
        additional_inputs: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            models: path to trained models or list of preloaded model
            neighbor_list: SchNetPack neighbor list
            energy_key: name of energies in model (default="energy")
            force_key: name of forces in model (default="forces")
            stress_key: name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit: energy units used by model (default="kcal/mol")
            position_unit: position units used by model (default="Angstrom")
            device: device used for calculations (default="cpu")
            input_dtype: select model input precision (default=float32)
            output_dtype: select model output precision (default=float64)
            converter: converter used to set up input batches
            transforms: transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            uncertainty_fn: Function to compute uncertainty. If not provided, defaults to AbsoluteUncertainty.
            additional_inputs: Additional arguments for basic ase calculator class
        """
        # Initialize the parent class without loading a model
        Calculator.__init__(self, **kwargs)

        self.neighbor_list = deepcopy(neighbor_list)
        self.device = device
        self.dtype = dtype
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

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy: self.energy_conversion,
            self.forces: self.energy_conversion / self.position_conversion,
            self.stress: self.energy_conversion / self.position_conversion**3,
        }

        # Load multiple models
        self.models = nn.ModuleList(
            [
                (
                    model
                    if isinstance(model, nn.Module)
                    else self._load_model(model, device, dtype)
                )
                for model in models
            ]
        ).to(dtype=self.dtype, device=self.device)

        # define uncertainty function as list
        if uncertainty_fn is None:
            uncertainty_fn = [AbsoluteUncertainty()]
        if not isinstance(uncertainty_fn, list):
            uncertainty_fn = [uncertainty_fn]
        self.uncertainty_fn = uncertainty_fn

    def calculate(
        self,
        atoms: Atoms = None,
        # properties is just a placeholder and will be ignored
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        properties = [
            p_key for p_key, p_value in self.property_map.items() if p_value is not None
        ]

        # check if calculation is needed
        if not self.calculation_required(atoms, properties):
            return

        # prepare calculation
        Calculator.calculate(self, atoms)
        model_inputs = self.converter(atoms)

        # calculate properties
        accumulated_results = {prop: [] for prop in properties}
        for model in self.models:
            model_results = model({p: model_inputs[p].clone() for p in model_inputs})
            for prop in properties:
                model_prop = self.property_map[prop]
                if model_prop in model_results:
                    # extract predictions in correct shape
                    value = model_results[model_prop].cpu().data.numpy()
                    if prop == self.energy:
                        value = value.item()
                    elif prop == self.stress:
                        value = value.squeeze()
                    # accumulate results
                    accumulated_results[prop].append(value * self.property_units[prop])
                else:
                    raise AtomsConverterError(
                        f"'{prop}' is not a property of your models. Please check the model properties!"
                    )

        # Compute average values
        accumulated_results = {
            prop: np.stack(value) for prop, value in accumulated_results.items()
        }
        self.results = {
            prop: np.mean(accumulated_results[prop], axis=0) for prop in properties
        }

        # Compute uncertainty using assigned uncertainty function
        # self.results["uncertainty"] = self.uncertainty_fn(accumulated_results)
        if len(self.uncertainty_fn) == 1:
            self.results["uncertainty"] = self.uncertainty_fn[0](accumulated_results)
        else:
            self.results["uncertainty"] = {
                type(fn).__name__: float(fn(accumulated_results))
                for fn in self.uncertainty_fn
            }

    def get_uncertainty(self, atoms):
        """
        Ensure calculation is up to date and return the uncertainty.
        """
        self.calculate(atoms)
        return self.results["uncertainty"]


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    """

    def __init__(
        self,
        molecule_path: str,
        working_dir: str,
        model_file: str,
        neighbor_list: Transform,
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
        transforms: Union[Transform, List[Transform]] = None,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            molecule_path: Path to initial geometry
            working_dir: Path to directory where files should be stored
            model_file: path to trained model
            neighbor_list: SchNetPack neighbor list
            energy_key: name of energies in model (default="energy")
            force_key: name of forces in model (default="forces")
            stress_key: name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit: energy units used by model (default="kcal/mol")
            position_unit: position units used by model (default="Angstrom")
            device: device used for calculations (default="cpu")
            dtype: select model precision (default=float32)
            converter: converter used to set up input batches
            optimizer_class: ASE optimizer used for structure relaxation.
            fixed_atoms: list of indices corresponding to atoms with positions fixed in space.
            transforms: transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            additional_inputs: additional inputs required for some transforms in the converter.
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
            model=model_file,
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

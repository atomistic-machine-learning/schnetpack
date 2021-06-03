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

from schnetpack import structure
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CastTo32, ASENeighborList, CastTo64
from schnetpack.units import convert_units

from typing import Optional, List, Union

log = logging.getLogger(__name__)

__all__ = ["SpkCalculator", "AseInterface"]


class SpkCalculatorError(Exception):
    pass


class SpkCalculator(Calculator):
    """
    ASE calculator for schnetpack machine learning models.

    Args:
        model (schnetpack.AtomisticModel): Trained model for calculations
        cutoff (float): environment cutoff used in calculator
        neighbor_list (schnetpack.transform.Transform, optional): neighbor list for computing interatomic distances.
        device (str, optional): select to run calculations on 'cuda' or 'cpu'
        energy (str, optional): name of energy property in provided model.
        forces (str, optional): name of forces in provided model.
        stress (str, optional): name of stress property in provided model.
        energy_units (str, optional): energy units used by model
        forces_units (str, optional): force units used by model
        stress_units (str, optional): stress units used by model
        precision (str): toggle model precision
        **kwargs: Additional arguments for basic ase calculator class
    """

    energy = "energy"
    forces = "forces"
    stress = "stress"
    implemented_properties = [energy, forces, stress]

    def __init__(
        self,
        model: schnetpack.model.AtomisticModel,
        cutoff: float,
        neighbor_list: schnetpack.transform.Transform = ASENeighborList,
        device: Union[str, torch.device] = "cpu",
        energy: str = "energy",
        forces: str = "forces",
        stress: str = "stress",
        energy_units: Union[str, float] = "kcal/mol",
        forces_units: Union[str, float] = "kcal/mol/Angstrom",
        stress_units: Union[str, float] = "kcal/mol/Angstrom/Angstrom/Angstrom",
        precision: str = "float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        self.device = device
        self.model = model
        self.model.to(device)

        # get transforms and initialize neighbor list
        self.transforms: List[schnetpack.transform.Transform] = [neighbor_list(cutoff)]
        # check if triple indices for angles need to be computed
        self._check_for_triples()

        # Set numerical precision
        if precision == "float32":
            self.model.to(torch.float32)
            self.transforms.append(CastTo32())
        elif precision == "float64":
            self.model.to(torch.float64)
            self.transforms.append(CastTo64())
        else:
            raise SpkCalculatorError(f"Unrecognized precision {precision}")

        # Set transforms to pre mode
        self._set_pre_mode_transforms()
        # TODO: activate computation of stress in model if requested

        # Mapping between ASE names and model outputs
        self.property_map = {
            self.energy: energy,
            self.forces: forces,
            self.stress: stress,
        }

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy: convert_units(energy_units, "eV"),
            self.forces: convert_units(forces_units, "eV/Angstrom"),
            self.stress: convert_units(stress_units, "eV/A/A/A"),
        }

    def _set_pre_mode_transforms(self):
        """
        Initialize preprocessor mode for transforms.
        """
        for t in self.transforms:
            t.preprocessor()

    def _atoms2input(self, atoms: ase.Atoms):
        """
        Convert ASE atoms object to input batch format for model prediction.

        Args:
            atoms (ase.Atoms): ASE atoms object of the molecule.

        Returns:
            dict[str, torch.Tensor]: input batch for model.
        """
        inputs = {
            structure.n_atoms: torch.tensor([atoms.get_global_number_of_atoms()]),
            structure.Z: torch.from_numpy(atoms.get_atomic_numbers()),
            structure.R: torch.from_numpy(atoms.get_positions()),
            structure.cell: torch.from_numpy(atoms.get_cell().array),
            structure.pbc: torch.from_numpy(atoms.get_pbc()),
        }

        for transform in self.transforms:
            inputs = transform(inputs)

        inputs = _atoms_collate_fn([inputs])

        # Move input batch to device
        inputs = {p: inputs[p].to(self.device) for p in inputs}

        return inputs

    def _check_for_triples(self):
        """
        Check, whether model representation requires collection of atomic triples and activate automatically.
        """
        if isinstance(
            self.model.representation,
            schnetpack.representation.symfuncs.SymmetryFunctions,
        ):
            if self.model.representation.n_basis_angular > 0:
                log.info("Enabling collection of atom triples for angular functions...")
                self.transforms.append(
                    schnetpack.transform.neighborlist.CollectAtomTriples()
                )

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)

            # Convert to schnetpack input format
            model_inputs = self._atoms2input(atoms)
            model_results = self.model(model_inputs)

            results = {}
            # TODO: use index information to slice everything properly
            for prop in properties:
                model_prop = self.property_map[prop]
                if model_prop in model_results:
                    if prop == self.energy:
                        # ase calculator should return scalar energy
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy()[0]
                            * self.property_units[prop]
                        )
                    else:
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy()
                            * self.property_units[prop]
                        )
                else:
                    raise SpkCalculatorError(
                        "'{:s}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(model_prop)
                    )

            self.results = results


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    """

    def __init__(
        self,
        molecule_path: str,
        working_dir: str,
        model: schnetpack.model.AtomisticModel,
        cutoff: float,
        neighbor_list: schnetpack.transform.Transform = ASENeighborList,
        device: Union[str, torch.device] = torch.device("cuda"),
        energy: str = "energy",
        forces: str = "forces",
        stress: str = "stress",
        energy_units: Union[str, float] = "kcal/mol",
        forces_units: Union[str, float] = "kcal/mol/Angstrom",
        stress_units: Union[str, float] = "kcal/mol/Angstrom/Angstrom/Angstrom",
        precision: str = "float64",
    ):
        """
        Args:
            molecule_path (str): Path to initial geometry
            working_dir (str): Path to directory where files should be stored
            model (schnetpack.model.AtomisticModel): Trained model
            cutoff (float): environment cutoff used in calculator
            neighbor_list (schnetpack.transform.Transform, optional): neighbor list for computing interatomic distances.
            device (str, optional): select to run calculations on 'cuda' or 'cpu'
            energy (str, optional): name of energy property in provided model.
            forces (str, optional): name of forces in provided model.
            stress (str, optional): name of stress property in provided model.
            energy_units (str, optional): energy units used by model
            forces_units (str, optional): force units used by model
            stress_units (str, optional): stress units used by model
            precision (str): toggle model precision
        """
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = read(molecule_path)

        # Set up calculator
        calculator = SpkCalculator(
            model,
            cutoff,
            neighbor_list=neighbor_list,
            device=device,
            energy=energy,
            forces=forces,
            stress=stress,
            energy_units=energy_units,
            forces_units=forces_units,
            stress_units=stress_units,
            precision=precision,
        )

        self.molecule.set_calculator(calculator)

        self.dynamics = None

    def save_molecule(self, name: str, file_format: str = "xyz", append: bool = False):
        """
        Save the current molecular geometry.

        Args:
            name (str): Name of save-file.
            file_format (str): Format to store geometry (default xyz).
            append (bool): If set to true, geometry is added to end of file
                (default False).
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

    def run_md(self, steps: int):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.

        Args:
            steps (int): Number of simulation steps performed
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
            fmax (float): Maximum residual force change (default 1.e-2)
            steps (int): Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(
            self.molecule,
            trajectory="{:s}.traj".format(optimize_file),
            restart="{:s}.pkl".format(optimize_file),
        )
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name)

    def compute_normal_modes(self, write_jmol: bool = True):
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

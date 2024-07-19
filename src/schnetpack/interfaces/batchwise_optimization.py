from copy import deepcopy
import os
import pickle
import time

import ase
import numpy as np
from math import sqrt
from os.path import isfile

from ase.optimize.optimize import Dynamics
from ase.parallel import world, barrier
from ase.io import write
from ase import Atoms

from typing import Dict, Optional, List, Tuple

import torch
from torch import nn
from schnetpack.units import convert_units
from schnetpack.interfaces.ase_interface import AtomsConverter


__all__ = [
    "ASEBatchwiseLBFGS",
    "BatchwiseCalculator",
    "BatchwiseEnsembleCalculator",
    "NNEnsemble",
]


class AtomsConverterError(Exception):
    pass


class NNEnsemble(nn.Module):
    # TODO: integrate this into EnsembleCalculator directly
    def __init__(self, models: nn.ModuleList, properties: List[str]):
        super(NNEnsemble, self).__init__()
        self.models = models
        if type(properties) == str:
            properties = [properties]
        self.properties = properties

    def setup(self, stage: Optional[str] = None) -> None:
        for model in self.models:
            model.setup(stage)

    def forward(
        self,
        x: Dict,
    ) -> Tuple:
        results = {}
        for p in self.properties:
            results[p] = []

        inputs = deepcopy(x)
        for model in self.models:
            x = deepcopy(inputs)
            predictions = model(x)
            for prop, values in predictions.items():
                if prop in self.properties:
                    results[prop].append(values)

        means = {}
        stds = {}
        for prop, values in results.items():
            stacked_values = torch.stack(values)
            means[prop] = stacked_values.mean(dim=0)
            stds[prop] = stacked_values.std(dim=0)

        return means, stds


class BatchwiseCalculator:
    """
    Calculator for neural network models for batchwise optimization.
    """

    def __init__(
        self,
        model: nn.Module or str,
        atoms_converter: AtomsConverter,
        device: str or torch.device = "cpu",
        auxiliary_output_modules: Optional[List] = None,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):
        """
        model:
            path to trained model or trained model

        atoms_converter:
            Class used to convert ase Atoms objects to schnetpack input

        device:
            device used for calculations (default="cpu")

        auxiliary_output_modules:
            auxiliary module to manipulate output properties (e.g., prior energy or forces)

        energy_key:
            name of energies in model (default="energy")

        force_key:
            name of forces in model (default="forces")

        stress_key:
            name of stress in model (default=None)

        energy_unit:
            energy units used by model (default="eV")

        position_unit:
            position units used by model (default="Angstrom")

        dtype:
            required data type for the model input (default: torch.float32)
        """

        self.results = None
        self.atoms = None

        if type(device) == str:
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.atoms_converter = atoms_converter
        self.auxiliary_output_modules = auxiliary_output_modules or []

        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy_key: self.energy_conversion,
            self.force_key: self.energy_conversion / self.position_conversion,
        }
        if self.stress_key is not None:
            self.property_units[self.stress_key] = (
                self.energy_conversion / self.position_conversion**3
            )

        # load model from path if needed
        if type(model) == str:
            model = self._load_model(model)

        self._initialize_model(model)

    def _load_model(self, model: str) -> nn.Module:
        return torch.load(model, map_location="cpu").to(torch.float64)

    def _initialize_model(self, model: nn.Module) -> None:
        for auxiliary_output_module in self.auxiliary_output_modules:
            model.output_modules.insert(1, auxiliary_output_module)
        self.model = model.eval()
        self.model.to(device=self.device, dtype=self.dtype)

    def _requires_calculation(self, property_keys: List[str], atoms: List[ase.Atoms]):
        if self.results is None:
            return True
        for name in property_keys:
            if name not in self.results:
                return True
        if len(self.atoms) != len(atoms):
            return True
        for atom, atom_ref in zip(atoms, self.atoms):
            if atom != atom_ref:
                return True

    def get_forces(
        self, atoms: List[ase.Atoms], fixed_atoms_mask: Optional[List[int]] = None
    ) -> np.array:
        """
        atoms:

        fixed_atoms_mask:
            list of indices corresponding to atoms with positions fixed in space.
        """
        if self._requires_calculation(
            property_keys=[self.energy_key, self.force_key], atoms=atoms
        ):
            self.calculate(atoms)
        f = self.results[self.force_key]
        if fixed_atoms_mask is not None:
            f[fixed_atoms_mask] = 0.0
        return f

    def get_potential_energy(self, atoms: List[ase.Atoms]) -> float:
        if self._requires_calculation(property_keys=[self.energy_key], atoms=atoms):
            self.calculate(atoms)
        return self.results[self.energy_key]

    def calculate(self, atoms: List[ase.Atoms]) -> None:
        property_keys = list(self.property_units.keys())
        inputs = self.atoms_converter(atoms)
        model_results = self.model(inputs)

        results = {}
        # store model results in calculator
        for prop in property_keys:
            if prop in model_results:
                results[prop] = (
                    model_results[prop].detach().cpu().numpy()
                    * self.property_units[prop]
                )
            else:
                raise AtomsConverterError(
                    "'{:s}' is not a property of your model. Please "
                    "check the model "
                    "properties!".format(prop)
                )

        self.results = results
        self.atoms = atoms.copy()


class BatchwiseEnsembleCalculator(BatchwiseCalculator):
    """
    Calculator for ensemble of neural network models for batchwise optimization.
    """

    # TODO: inherit from SpkEnsembleCalculator
    def __init__(
        self,
        model: str or nn.ModuleList,
        atoms_converter: AtomsConverter,
        device: str or torch.device = "cpu",
        auxiliary_output_modules: Optional[List[nn.Module]] = None,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):
        """
        model:
            Directory of trained models or module list of trained models

        atoms_converter:
            Class used to convert ase Atoms objects to schnetpack input

        device:
            device used for calculations (default="cpu")

        auxiliary_output_modules:
            auxiliary module to manipulate output properties (e.g., prior energy or forces)

        energy_key:
            name of energies in model (default="energy")

        force_key:
            name of forces in model (default="forces")

        energy_unit:
            energy units used by model (default="eV")

        stress_key:
            name of stress in model (default=None)

        position_unit:
            position units used by model (default="Angstrom")

        dtype:
            required data type for the model input (default: torch.float32)
        """
        super(BatchwiseEnsembleCalculator, self).__init__(
            model=model,
            atoms_converter=atoms_converter,
            device=device,
            auxiliary_output_modules=auxiliary_output_modules,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            dtype=dtype,
        )

    def _load_model(self, model: str) -> nn.ModuleList:
        # get model paths
        model_names = os.listdir(model)
        model_paths = [os.path.join(model, model_name) for model_name in model_names]

        # create module list
        models = torch.nn.ModuleList()
        for m_path in model_paths:
            m = torch.load(os.path.join(m_path, "best_model"), map_location="cpu").to(
                torch.float64
            )
            models.append(m)

        return models

    def _initialize_model(self, model: nn.ModuleList) -> None:
        # add auxiliary output modules
        for m in model:
            for auxiliary_output_module in self.auxiliary_output_modules:
                m.output_modules.insert(1, auxiliary_output_module)

        # initialize ensemble
        ensemble = NNEnsemble(models=model, properties=list(self.property_units.keys()))
        self.model = ensemble.eval().to(device=self.device, dtype=self.dtype)

    def calculate(self, atoms: List[ase.Atoms]) -> None:
        property_keys = list(self.property_units.keys())
        inputs = self.atoms_converter(atoms)
        model_results, stds = self.model(inputs)

        results = {}
        # store model uncertainties in calculator
        for prop in property_keys:
            if prop in model_results:
                results["{}_uncertainty".format(prop)] = (
                    stds[prop].detach().cpu().numpy() * self.property_units[prop]
                )

        # store model results in calculator
        for prop in property_keys:
            if prop in model_results:
                results[prop] = (
                    model_results[prop].detach().cpu().numpy()
                    * self.property_units[prop]
                )
            else:
                raise AtomsConverterError(
                    "'{:s}' is not a property of your model. Please "
                    "check the model "
                    "properties!".format(prop)
                )

        self.results = results
        self.atoms = atoms.copy()


class BatchwiseDynamics(Dynamics):
    """Base-class for batch-wise MD and structure optimization classes."""

    def __init__(
        self,
        calculator: BatchwiseCalculator,
        atoms: List[Atoms],
        logfile: str,
        trajectory: Optional[str],
        append_trajectory: bool = False,
        master: Optional[bool] = None,
        log_every_step: bool = False,
        fixed_atoms_mask: Optional[List[int]] = None,
    ):
        """Structure dynamics object.

        Parameters:

        calculator:
            This calculator provides properties such as forces and energy, which can be used for MD simulations or
            relaxations

        atoms:
            The Atoms objects to relax.

        restart:
            Filename for restart file.  Default value is *None*.

        logfile:
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory:
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        append_trajectory:
            Appended to the trajectory file instead of overwriting it.

        master:
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        log_every_step:
            set to True to log Dynamics after each step (default=False)

        fixed_atoms:
            list of indices corresponding to atoms with positions fixed in space.
        """
        super().__init__(
            atoms=atoms,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            master=master,
        )

        self.calculator = calculator
        self.trajectory = trajectory
        self.log_every_step = log_every_step
        self.fixed_atoms_mask = fixed_atoms_mask
        self.n_configs = len(self.atoms)
        self.n_atoms = len(self.atoms[0])

    def irun(self):
        # compute initial structure and log the first step
        self.calculator.get_forces(self.atoms, fixed_atoms_mask=self.fixed_atoms_mask)

        # yield the first time to inspect before logging
        yield False

        if self.nsteps == 0:
            self.log()
            pass

        # run the algorithm until converged or max_steps reached
        while not self.converged() and self.nsteps < self.max_steps:

            # compute the next step
            self.step()
            self.nsteps += 1

            # let the user inspect the step and change things before logging
            # and predicting the next step
            yield False

            # log the step
            if self.log_every_step:
                self.log()

        # log last step
        self.log()

        # finally check if algorithm was converged
        yield self.converged()

    def run(self) -> bool:
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        for converged in BatchwiseDynamics.irun(self):
            pass
        return converged


class BatchwiseOptimizer(BatchwiseDynamics):
    """Base-class for all structure optimization classes."""

    # default maxstep for all optimizers
    defaults = {"maxstep": 0.2}

    def __init__(
        self,
        calculator: BatchwiseCalculator,
        atoms: List[Atoms],
        restart: Optional[bool] = None,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        master: Optional[str] = None,
        append_trajectory: bool = False,
        log_every_step: bool = False,
        fixed_atoms_mask: Optional[List[int]] = None,
    ):
        """Structure optimizer object.

        Parameters:

        calculator:
            This calculator provides properties such as forces and energy, which can be used for MD simulations or
            relaxations

        atoms:
            The Atoms objects to relax.

        restart:
            Filename for restart file.  Default value is *None*.

        logfile:
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory:
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        master:
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        append_trajectory:
            Appended to the trajectory file instead of overwriting it.

        log_every_step:
            set to True to log Dynamics after each step (default=False)

        fixed_atoms:
            list of indices corresponding to atoms with positions fixed in space.
        """
        BatchwiseDynamics.__init__(
            self,
            calculator=calculator,
            atoms=atoms,
            logfile=logfile,
            trajectory=trajectory,
            master=master,
            append_trajectory=append_trajectory,
            log_every_step=log_every_step,
            fixed_atoms_mask=fixed_atoms_mask,
        )

        self.restart = restart

        # initialize attribute
        self.fmax = None

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            barrier()

    def todict(self) -> Dict:
        description = {
            "type": "optimization",
            "optimizer": self.__class__.__name__,
        }
        return description

    def initialize(self):
        pass

    def irun(self, fmax: float = 0.05, steps: Optional[int] = None):
        """call Dynamics.irun and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return BatchwiseDynamics.irun(self)

    def run(self, fmax: float = 0.05, steps: Optional[int] = None):
        """call Dynamics.run and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return BatchwiseDynamics.run(self)

    def converged(self, forces: Optional[np.array] = None) -> bool:
        """Did the optimization converge?"""
        if forces is None:
            forces = self.calculator.get_forces(
                self.atoms, fixed_atoms_mask=self.fixed_atoms_mask
            )
        # todo: maybe np.linalg.norm?
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def log(self, forces: Optional[np.array] = None) -> None:
        if forces is None:
            forces = self.calculator.get_forces(
                self.atoms, fixed_atoms_mask=self.fixed_atoms_mask
            )
        fmax = sqrt((forces**2).sum(axis=1).max())
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "fmax")
                msg = "%s  %4s %8s %12s\n" % args
                self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], fmax)
            msg = "%s:  %3d %02d:%02d:%02d %12.4f\n" % args
            self.logfile.write(msg)

            self.logfile.flush()

        if self.trajectory is not None:
            for struc_idx, at in enumerate(self.atoms):
                # store in trajectory
                write(
                    self.trajectory + "_{}.xyz".format(struc_idx),
                    at,
                    format="extxyz",
                    append=False if self.nsteps == 0 else True,
                )

    def get_relaxation_results(self) -> Tuple[Atoms, Dict]:
        self.calculator.get_forces(self.atoms)
        return self.atoms, self.calculator.results

    def dump(self, data):
        if world.rank == 0 and self.restart is not None:
            with open(self.restart, "wb") as fd:
                pickle.dump(data, fd, protocol=2)

    def load(self):
        with open(self.restart, "rb") as fd:
            return pickle.load(fd)


class ASEBatchwiseLBFGS(BatchwiseOptimizer):
    """Limited memory BFGS optimizer.

    LBFGS optimizer that allows for relaxation of multiple structures in parallel. This optimizer is an
    extension/adaptation of the ase.optimize.LBFGS optimizer particularly designed for batch-wise relaxation
    of atomic structures. The inverse Hessian is approximated for each sample separately, which allows for
    optimizing batches of different structures/compositions.

    """

    def __init__(
        self,
        calculator: BatchwiseCalculator,
        atoms: List[Atoms],
        restart: Optional[bool] = None,
        logfile: str = "-",
        trajectory: Optional[str] = None,
        maxstep: Optional[float] = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        use_line_search: bool = False,
        master: Optional[str] = None,
        log_every_step: bool = False,
        fixed_atoms_mask: Optional[List[int]] = None,
        verbose: bool = False,
    ):
        """Parameters:

        calculator:
            This calculator provides properties such as forces and energy, which can be used for MD simulations or
            relaxations

        atoms:
            The Atoms objects to relax.

        restart:
            Pickle file used to store vectors for updating the inverse of
            Hessian matrix. If set, file with such a name will be searched
            and information stored will be used, if the file exists.

        logfile:
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory:
            Pickle file used to store trajectory of atomic movement.

        maxstep:
            How far is a single atom allowed to move. This is useful for DFT
            calculations where wavefunctions can be reused if steps are small.
            Default is 0.2 Angstrom.

        memory:
            Number of steps to be stored. Default value is 100. Three numpy
            arrays of this length containing floats are stored.

        damping:
            The calculated step is multiplied with this number before added to
            the positions.

        alpha:
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        use_line_search:
            Not implemented yet.

        master:
            Defaults to None, which causes only rank 0 to save files.  If
            set to true, this rank will save files.

        log_every_step:
            set to True to log Dynamics after each step (default=False)

        fixed_atoms:
            list of indices corresponding to atoms with positions fixed in space.
        """

        BatchwiseOptimizer.__init__(
            self,
            calculator=calculator,
            atoms=atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            master=master,
            log_every_step=log_every_step,
            fixed_atoms_mask=fixed_atoms_mask,
        )

        if maxstep is not None:
            self.maxstep = maxstep
        else:
            self.maxstep = self.defaults["maxstep"]

        if self.maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for "
                + "the maximum step size: %.1f Angstrom" % maxstep
            )

        self.memory = memory
        # Initial approximation of inverse Hessian 1./70. is to emulate the
        # behaviour of BFGS. Note that this is never changed!
        self.H0 = 1.0 / alpha
        self.damping = damping
        self.use_line_search = use_line_search
        self.p = None
        self.function_calls = 0
        self.force_calls = 0
        self.n_normalizations = 0

        self.verbose = verbose

        if use_line_search:
            raise NotImplementedError("Lines search has not been implemented yet")

    def initialize(self) -> None:
        """Initialize everything so no checks have to be done in step"""
        self.iteration = 0
        self.s = []
        self.y = []
        # Store also rho, to avoid calculating the dot product again and
        # again.
        self.rho = []

        self.r0 = None
        self.f0 = None
        self.e0 = None
        self.task = "START"
        self.load_restart = False

    def read(self) -> None:
        """Load saved arrays to reconstruct the Hessian"""
        (
            self.iteration,
            self.s,
            self.y,
            self.rho,
            self.r0,
            self.f0,
            self.e0,
            self.task,
        ) = self.load()
        self.load_restart = True

    def step(self, f: np.array = None) -> None:
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it"""

        if f is None:
            f = self.calculator.get_forces(
                self.atoms, fixed_atoms_mask=self.fixed_atoms_mask
            )

        # check if updates for respective structures are required
        q_euclidean = -f.reshape(self.n_configs, -1, 3)
        squared_max_forces = (q_euclidean**2).sum(axis=-1).max(axis=-1)
        configs_mask = squared_max_forces < self.fmax**2
        mask = (
            configs_mask[:, None]
            .repeat(q_euclidean.shape[1], 0)
            .repeat(q_euclidean.shape[2], 1)
        )
        r = np.zeros((self.n_atoms * self.n_configs, 3), dtype=np.float64)
        for config_idx, at in enumerate(self.atoms):
            first_idx = config_idx * self.n_atoms
            last_idx = config_idx * self.n_atoms + self.n_atoms
            r[first_idx:last_idx] = at.get_positions()

        self.update(r, f, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty(
            (
                loopmax,
                self.n_configs,
                1,
                1,
            ),
            dtype=np.float64,
        )

        # ## The algorithm itself:
        q = -f.reshape(self.n_configs, 1, -1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * np.matmul(s[i], np.transpose(q, axes=(0, 2, 1)))
            q -= a[i] * y[i]

        z = H0 * q

        for i in range(loopmax):
            b = rho[i] * np.matmul(y[i], np.transpose(z, axes=(0, 2, 1)))
            z += s[i] * (a[i] - b)

        p = -z.reshape((-1, 3))
        self.p = np.where(mask, np.zeros_like(p), p)
        # ##

        g = -f
        if self.use_line_search is True:
            e = self.func(r)
            self.line_search(r, g, e)
            dr = (self.alpha_k * self.p).reshape(self.n_atoms * self.n_configs, -1)
        else:
            self.force_calls += 1
            self.function_calls += 1
            dr = self.determine_step(self.p) * self.damping

        # update positions
        pos_updated = r + dr

        # create new list of ase Atoms objects with updated positions
        ats = []
        for config_idx, at in enumerate(self.atoms):
            first_idx = config_idx * self.n_atoms
            last_idx = config_idx * self.n_atoms + self.n_atoms
            at = Atoms(
                positions=pos_updated[first_idx:last_idx],
                numbers=self.atoms[config_idx].get_atomic_numbers(),
            )
            at.pbc = self.atoms[config_idx].pbc
            at.cell = self.atoms[config_idx].cell
            ats.append(at)
        self.atoms = ats

        self.iteration += 1
        self.r0 = r
        self.f0 = -g
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )

    def determine_step(self, dr: np.array) -> np.array:
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        steplengths = (dr**2).sum(-1) ** 0.5
        # check if any step in entire batch is greater than maxstep
        if np.max(steplengths) >= self.maxstep:
            # rescale steps for each config separately
            for config_idx in range(self.n_configs):
                first_idx = config_idx * self.n_atoms
                last_idx = config_idx * self.n_atoms + self.n_atoms
                longest_step = np.max(steplengths[first_idx:last_idx])
                if longest_step >= self.maxstep:
                    if self.verbose:
                        print("normalized integration step")
                    self.n_normalizations += 1
                    dr[first_idx:last_idx] *= self.maxstep / longest_step
        return dr

    def update(self, r: np.array, f: np.array, r0: np.array, f0: np.array) -> None:
        """Update everything that is kept in memory

        This function is mostly here to allow for replay_trajectory.
        """
        if self.iteration > 0:
            s0 = r.reshape(self.n_configs, 1, -1) - r0.reshape(self.n_configs, 1, -1)
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0.reshape(self.n_configs, 1, -1) - f.reshape(self.n_configs, 1, -1)
            self.y.append(y0)

            rho0 = np.ones((self.n_configs, 1, 1), dtype=np.float64)
            for config_idx in range(self.n_configs):
                ys0 = np.dot(y0[config_idx, 0], s0[config_idx, 0])
                if ys0 > 1e-8:
                    rho0[config_idx, 0, 0] = 1.0 / ys0
            self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def func(self, x):
        """Objective function for use of the optimizers"""
        raise NotImplementedError("func not implemented yet")

    def line_search(self, r, g, e):
        self.alpha_k = None
        raise NotImplementedError("LineSearch not implemented yet")

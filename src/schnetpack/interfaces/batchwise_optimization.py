"""
Batch-wise structure relaxation for SchNetPack models.

``BatchwiseLBFGS`` relaxes a whole batch of structures in parallel, keeping one inverse
Hessian approximation per structure so that batches of differing compositions can be
optimized together.

The batch is a SchNetPack input dictionary of torch tensors and stays that way for the
entire run -- this module contains no ase code at all. Trajectories go to a single
buffered HDF5 file (see :mod:`schnetpack.interfaces.batchwise_trajectory`), and callers
convert at the boundary with :func:`~schnetpack.interfaces.ase_interface.atoms_to_batch`
on the way in and :func:`~schnetpack.interfaces.ase_interface.batch_to_atoms` on the way
out.

Note:
    ``BatchwiseEnsembleCalculator`` and ``NNEnsemble`` have not been migrated to the
    tensor-based calculator contract. ``BatchwiseEnsembleCalculator.calculate`` still
    expects a list of ``ase.Atoms`` and returns numpy arrays, so the inherited
    ``get_forces(inputs)`` raises. They are kept for backwards compatibility, are not
    exercised by any test, and are the only ase-shaped thing left in this module.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import ExitStack
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from schnetpack import properties
from schnetpack.interfaces.batchwise_trajectory import BatchwiseTrajectoryWriter
from schnetpack.units import convert_units
from schnetpack.utils.compatibility import load_model

from schnetpack.transform import BatchNeighborList

if TYPE_CHECKING:  # import only for type checking, so the runtime stays ase-free
    from ase import Atoms

__all__ = [
    "BatchwiseLBFGS",
    "BatchwiseCalculator",
    "BatchwiseCalculatorError",
    "BatchwiseEnsembleCalculator",
    "BatchwiseOptimizer",
    "NNEnsemble",
]


class BatchwiseCalculatorError(Exception):
    pass


class NNEnsemble(nn.Module):
    # TODO: integrate this into EnsembleCalculator directly
    def __init__(self, models: nn.ModuleList, properties: List[str]):
        super(NNEnsemble, self).__init__()
        self.models = models
        if isinstance(properties, str):
            properties = [properties]
        self.properties = properties

    def setup(self, stage: Optional[str] = None) -> None:
        for model in self.models:
            model.setup(stage)

    def forward(self, x: Dict) -> Tuple:
        results = {p: [] for p in self.properties}

        inputs = deepcopy(x)
        for model in self.models:
            predictions = model(deepcopy(inputs))
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
    """Evaluates a SchNetPack model on a whole batch of structures at once.

    Results are cached and only recomputed when the structure actually changed, so a
    relaxation pays for exactly one model call per step even though the optimizer asks
    for forces several times per step.

    Args:
        model: trained model, or a path to one. The calculator evaluates it as it is
            given -- to add a prior to the energy, compose it into the model's output
            modules beforehand (see ``examples/howtos/howto_batchwise_relaxations.ipynb``).
        neighbor_list: keeps the batch's neighbor lists valid as the structures move.
            Most steps reuse the previous list rather than rebuilding it, which is a
            large part of why relaxing a batch pays off.
        device: device the model runs on.
        energy_key, force_key, stress_key: names of these properties in the model.
            ``stress_key=None`` disables stress.
        energy_unit, position_unit: units the model works in. Results are converted to
            ase units (eV, Angstrom).
        dtype: precision of the model input.
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        neighbor_list: BatchNeighborList,
        device: Union[str, torch.device] = "cpu",
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):
        self.results = None
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.neighbor_list = neighbor_list

        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # unit conversion to default ase units
        energy_conversion = convert_units(energy_unit, "eV")
        position_conversion = convert_units(position_unit, "Angstrom")
        self.property_units = {
            energy_key: energy_conversion,
            force_key: energy_conversion / position_conversion,
        }
        if stress_key is not None:
            self.property_units[stress_key] = energy_conversion / position_conversion**3

        if isinstance(model, str):
            model = self._load_model(model)
        self._initialize_model(model)

        # the structure self.results was computed for
        self._cached_structure = None

    def _load_model(self, model: str) -> nn.Module:
        return load_model(model, device="cpu").to(torch.float64)

    def _initialize_model(self, model: nn.Module) -> None:
        self.model = model.eval()
        self.model.to(device=self.device, dtype=self.dtype)

    #: input entries that decide whether a cached result is still valid
    _structure_keys = (properties.R, properties.cell, properties.pbc)

    def _structure_id(self, inputs: Dict[str, torch.Tensor]) -> Tuple:
        """Fingerprint of the structure the inputs describe.

        Pairs each tensor with ``_version``, the counter autograd bumps on in-place
        mutation, so both a rebound entry and an updated one are noticed. Unlike an
        element-wise comparison this never synchronizes with the device, which matters
        because it is checked on every force request. Keeping the tensors themselves in
        the fingerprint also keeps them alive, so a freed tensor cannot be mistaken for
        the cached one.
        """
        return tuple(
            (inputs[key], inputs[key]._version) for key in self._structure_keys
        )

    def _requires_calculation(
        self, property_keys: List[str], inputs: Dict[str, torch.Tensor]
    ) -> bool:
        if self.results is None or self._cached_structure is None:
            return True
        if any(name not in self.results for name in property_keys):
            return True
        return any(
            inputs[key] is not tensor or tensor._version != version
            for (tensor, version), key in zip(
                self._cached_structure, self._structure_keys
            )
        )

    def get_forces(
        self,
        inputs: Dict[str, torch.Tensor],
        fixed_atoms_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forces on every atom of the batch, in eV/Angstrom.

        Args:
            inputs: schnetpack input batch.
            fixed_atoms_mask: boolean mask selecting the atoms to return forces for.
                Defaults to all of them.
        """
        if self._requires_calculation([self.energy_key, self.force_key], inputs):
            self.calculate(inputs)
        forces = self.results[self.force_key]
        return forces if fixed_atoms_mask is None else forces[fixed_atoms_mask]

    def get_potential_energy(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Potential energy of every structure of the batch, in eV."""
        if self._requires_calculation([self.energy_key], inputs):
            self.calculate(inputs)
        return self.results[self.energy_key]

    def calculate(self, inputs: Dict[str, torch.Tensor]) -> None:
        structure_id = self._structure_id(inputs)

        # Shallow copy: the update replaces entries (neighbor lists, casts) and must not
        # write them back into the caller's batch, but the tensors themselves are only
        # read, so there is no reason to copy them all -- deep copying the batch would
        # mean copying the neighbor list on every single step.
        # Positions are the exception: the model marks them as requiring grad to get the
        # forces, and would otherwise turn the optimizer's live positions into a leaf
        # variable that can no longer be updated in place.
        inputs = dict(inputs)
        inputs[properties.R] = inputs[properties.R].detach().clone()
        inputs = self.neighbor_list.update(inputs)

        model_results = self.model(inputs)

        results = {}
        for prop, unit in self.property_units.items():
            if prop not in model_results:
                raise BatchwiseCalculatorError(
                    f"'{prop}' is not a property of your model. "
                    "Please check the model properties!"
                )
            results[prop] = model_results[prop].detach() * unit

        self.results = results
        self._cached_structure = structure_id


class BatchwiseEnsembleCalculator(BatchwiseCalculator):
    """Calculator for an ensemble of models, reporting per-property uncertainties.

    Warning:
        Not migrated to the tensor-based calculator contract, see the module docstring.
        ``calculate`` below still expects a list of ``ase.Atoms`` and returns numpy, so
        the inherited ``get_forces(inputs)`` raises.

    Args:
        model: directory of trained models, or a module list of them. Remaining
            arguments are those of :class:`BatchwiseCalculator`.
    """

    # TODO: inherit from SpkEnsembleCalculator
    def _load_model(self, model: str) -> nn.ModuleList:
        models = torch.nn.ModuleList()
        for model_name in os.listdir(model):
            models.append(
                load_model(
                    os.path.join(model, model_name, "best_model"), device="cpu"
                ).to(torch.float64)
            )
        return models

    def _initialize_model(self, model: nn.ModuleList) -> None:
        ensemble = NNEnsemble(models=model, properties=list(self.property_units.keys()))
        self.model = ensemble.eval().to(device=self.device, dtype=self.dtype)

    def calculate(self, atoms: List["Atoms"]) -> None:
        from schnetpack.interfaces.ase_interface import atoms_to_batch

        inputs = self.neighbor_list.update(
            atoms_to_batch(atoms, device=self.device, dtype=self.dtype)
        )
        model_results, stds = self.model(inputs)

        results = {}
        for prop, unit in self.property_units.items():
            if prop not in model_results:
                raise BatchwiseCalculatorError(
                    f"'{prop}' is not a property of your model. "
                    "Please check the model properties!"
                )
            results[prop] = model_results[prop].detach().cpu().numpy() * unit
            results[f"{prop}_uncertainty"] = stds[prop].detach().cpu().numpy() * unit

        self.results = results
        self.atoms = atoms.copy()


class BatchwiseOptimizer(ABC):
    """Drives a batch of structures downhill until every one of them is relaxed.

    Subclasses supply :meth:`step`; everything else -- the run loop, the convergence
    criterion, the text log and the HDF5 trajectory -- lives here.

    Positions are updated in place, so ``inputs`` holds the relaxed structures once the
    run is over.

    Args:
        calculator: provides the forces and energies driving the relaxation.
        inputs: schnetpack input batch holding the structures to relax. All structures
            must have the same number of atoms.
        logfile: text progress log. A path, ``"-"`` for stdout, or ``None`` for no log.
        log_interval: how often to write a log line. See below.
        trajectory: path of the HDF5 trajectory to write, or ``None`` for none.
        trajectory_interval: how often to write a trajectory frame. See below.
        store_forces: store the forces of every trajectory frame, not just the
            positions. Doubles the file size.
        fixed_atoms_mask: boolean mask over all atoms in the batch, True for atoms
            whose positions are held fixed in space.
        max_steps: step limit used when ``run`` is called without one.

    Both intervals count optimizer steps: ``0`` writes only the first and last frame,
    ``1`` writes every step, ``n`` writes every nth. The first and last frame are
    always written whatever the interval, and the two intervals are independent.
    """

    def __init__(
        self,
        calculator: BatchwiseCalculator,
        inputs: Dict[str, torch.Tensor],
        logfile: Optional[str] = None,
        log_interval: int = 1,
        trajectory: Optional[str] = None,
        trajectory_interval: int = 0,
        store_forces: bool = False,
        fixed_atoms_mask: Optional[List[bool]] = None,
        max_steps: int = 100_000,
    ):
        self.calculator = calculator
        self.inputs = inputs
        self.nsteps = 0
        self.max_steps = max_steps
        self.fmax = None
        # per-structure squared max force for the forces the last convergence check
        # saw, reused by step() and the loggers so the reduction is not repeated
        self._max_sq_force_per_config = None

        n_atoms = inputs[properties.n_atoms]
        self.n_configs = n_atoms.shape[0]
        if not bool((n_atoms == n_atoms[0]).all()):
            raise ValueError(
                "batch-wise optimization requires all structures in the batch to have "
                f"the same number of atoms, got {n_atoms.tolist()}"
            )
        self.n_atoms = int(n_atoms[0])

        # kept as a float mask rather than an index: zeroing the displacement of fixed
        # atoms is equivalent to dropping them from the optimization (their history
        # contributions are identically zero) and needs no device synchronization,
        # which boolean-mask indexing would force on every step
        n_total_atoms = self.n_configs * self.n_atoms
        device = inputs[properties.R].device
        if fixed_atoms_mask is None:
            self.free_atoms = torch.ones(
                (n_total_atoms, 1), dtype=torch.float64, device=device
            )
        else:
            fixed = torch.as_tensor(fixed_atoms_mask, dtype=torch.bool).view(-1, 1)
            if fixed.shape[0] != n_total_atoms:
                raise ValueError(
                    f"fixed_atoms_mask has {fixed.shape[0]} entries, expected one per "
                    f"atom in the batch ({n_total_atoms})"
                )
            self.free_atoms = (~fixed).to(dtype=torch.float64, device=device)

        self._closer = ExitStack()
        self.log_interval = log_interval
        if logfile is None:
            self.logfile = None
        elif logfile == "-":
            self.logfile = sys.stdout
        else:
            self.logfile = self._closer.enter_context(
                open(logfile, "a", encoding="utf-8")
            )

        self.trajectory = trajectory
        self.trajectory_interval = trajectory_interval
        self.store_forces = store_forces
        self._writer = None

    # ------------------------------------------------------------------ run loop

    @abstractmethod
    def step(self) -> None:
        """Move every structure of the batch one step downhill."""

    def max_squared_force_per_config(self, forces: torch.Tensor) -> torch.Tensor:
        """Largest squared force norm within each structure of the batch.

        Fixed atoms are excluded -- their residual force says nothing about whether the
        free atoms have relaxed.
        """
        squared = forces.view(self.n_configs, self.n_atoms, 3).pow(2).sum(-1)
        squared = squared * self.free_atoms.view(self.n_configs, self.n_atoms)
        return squared.max(-1).values

    def converged(self, forces: Optional[torch.Tensor] = None) -> bool:
        """Is every structure of the batch relaxed to within ``fmax``?"""
        if forces is None:
            forces = self.calculator.get_forces(self.inputs)
        self._max_sq_force_per_config = self.max_squared_force_per_config(forces)
        return bool((self._max_sq_force_per_config.max() < self.fmax**2).item())

    def irun(self, fmax: float = 0.05, steps: Optional[int] = None):
        """Drive the relaxation step by step, yielding after every step.

        The final value yielded is whether the batch converged.
        """
        self.fmax = fmax
        if steps is not None:
            self.max_steps = steps

        converged = False
        while True:
            # one model call per iteration; the loggers and step() below reuse it
            converged = self.converged()
            final = converged or self.nsteps >= self.max_steps
            self._write_frame(final=final)
            if final:
                break

            self.step()
            self.nsteps += 1
            # let the caller inspect the step before the next one is computed
            yield False

        yield converged

    def run(self, fmax: float = 0.05, steps: Optional[int] = None) -> bool:
        """Relax until converged or ``steps`` steps have been taken.

        Returns whether the maximum force on every free atom dropped below ``fmax``.
        """
        converged = False
        for converged in self.irun(fmax=fmax, steps=steps):
            pass
        return converged

    def get_relaxation_results(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """The relaxed batch and the model results for it.

        Both are the live tensors, at full precision and on the calculator's device.
        Use :func:`schnetpack.interfaces.ase_interface.batch_to_atoms` on the batch to
        get ``ase.Atoms`` back.
        """
        self.calculator.get_forces(self.inputs)
        return self.inputs, self.calculator.results

    # ------------------------------------------------------------------- logging

    def _is_due(self, interval: int, final: bool) -> bool:
        if final or self.nsteps == 0:
            return True
        return interval > 0 and self.nsteps % interval == 0

    def _write_frame(self, final: bool = False) -> None:
        """Write a log line and a trajectory frame, if this step calls for them."""
        log_now = self.logfile is not None and self._is_due(self.log_interval, final)
        trajectory_now = self.trajectory is not None and self._is_due(
            self.trajectory_interval, final
        )
        if not (log_now or trajectory_now):
            # nothing to write, so do not pay for the transfer off the device
            return

        forces = self.calculator.get_forces(self.inputs)
        if self._max_sq_force_per_config is None:
            self._max_sq_force_per_config = self.max_squared_force_per_config(forces)

        if log_now:
            self._write_log_line(self._max_sq_force_per_config.max().sqrt().item())
        if trajectory_now:
            self._write_trajectory_frame(forces)

    def _write_log_line(self, fmax: float) -> None:
        name = self.__class__.__name__
        if self.nsteps == 0:
            header = (" " * len(name), "Step", "Time", "fmax")
            self.logfile.write("%s  %4s %8s %12s\n" % header)

        clock = time.localtime()
        line = (name, self.nsteps, clock[3], clock[4], clock[5], fmax)
        self.logfile.write("%s:  %3d %02d:%02d:%02d %12.4f\n" % line)
        self.logfile.flush()

    def _write_trajectory_frame(self, forces: torch.Tensor) -> None:
        if self._writer is None:
            self._writer = self._closer.enter_context(
                BatchwiseTrajectoryWriter(
                    self.trajectory,
                    atomic_numbers=self.inputs[properties.Z],
                    pbc=self.inputs[properties.pbc],
                    store_forces=self.store_forces,
                    attrs={"optimizer": self.__class__.__name__, "fmax": self.fmax},
                )
            )
        self._writer.write(
            step=self.nsteps,
            positions=self.inputs[properties.R],
            cell=self.inputs[properties.cell],
            energy=self.calculator.get_potential_energy(self.inputs),
            forces=forces if self.store_forces else None,
            converged=self._max_sq_force_per_config < self.fmax**2,
        )

    def close(self) -> None:
        """Close the log file and the trajectory."""
        self._closer.close()
        self._writer = None

    def __enter__(self) -> "BatchwiseOptimizer":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class BatchwiseLBFGS(BatchwiseOptimizer):
    """Limited memory BFGS, relaxing a batch of structures in parallel.

    An adaptation of ``ase.optimize.LBFGS`` for batch-wise relaxation: the inverse
    Hessian is approximated for each structure separately, so batches of different
    structures and compositions can be optimized together.

    Args:
        maxstep: how far a single atom is allowed to move in one step, in Angstrom.
            Each structure of the batch is rescaled on its own.
        memory: number of steps of history kept for the two-loop recursion.
        damping: the calculated step is multiplied by this before it is taken.
        alpha: initial guess for the curvature of the energy surface. The conservative
            default of 70.0 emulates BFGS; a lower value may converge in fewer steps at
            the cost of stability.
        device: device the L-BFGS bookkeeping runs on (default: cpu). The two-loop
            recursion is bound by kernel launches rather than arithmetic -- it was
            measured 5-7x slower on cuda than on cpu for batches up to 256 structures
            of 1000 atoms -- so the default is cpu regardless of where the model runs.
            Worth re-measuring before overriding for much larger batches.

    Remaining keyword arguments are those of :class:`BatchwiseOptimizer`.
    """

    #: how far a single atom may move in one step, in Angstrom
    default_maxstep = 0.2

    def __init__(
        self,
        calculator: BatchwiseCalculator,
        inputs: Dict[str, torch.Tensor],
        maxstep: Optional[float] = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__(calculator=calculator, inputs=inputs, **kwargs)

        self.maxstep = self.default_maxstep if maxstep is None else maxstep
        if self.maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for the maximum step size: "
                f"{self.maxstep:.1f} Angstrom"
            )

        self.memory = memory
        # Initial approximation of the inverse Hessian, 1./70. to emulate the behaviour
        # of BFGS. Note that this is never changed!
        self.H0 = 1.0 / alpha
        self.damping = damping
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        # same mask as self.free_atoms, but on the device the recursion runs on
        self._free_atoms_opt = self.free_atoms.to(self.device)

        self.iteration = 0
        self.s = []
        self.y = []
        # rho is stored alongside, to avoid calculating the dot product again and again
        self.rho = []
        self.r0 = None
        self.f0 = None

    def step(
        self, f: Optional[torch.Tensor] = None, normalize_step: bool = True
    ) -> None:
        """Update the history, compute the next step, and take it."""
        if f is None:
            f = self.calculator.get_forces(self.inputs)
        # forces on fixed atoms are zeroed rather than removed: their history
        # contributions vanish, so the recursion below is unchanged by them
        f = f.to(device=self.device, dtype=torch.float64) * self._free_atoms_opt

        # structures that already meet the force criterion must not be moved further.
        # f is masked, so the fixed atoms drop out of the reduction on their own
        if self._max_sq_force_per_config is None:
            max_sq_force = f.view(self.n_configs, self.n_atoms, 3).pow(2).sum(-1)
            max_sq_force = max_sq_force.max(-1).values
        else:
            max_sq_force = self._max_sq_force_per_config.to(self.device)
        relaxed = max_sq_force < self.fmax**2

        r = self.inputs[properties.R].to(device=self.device, dtype=torch.float64)
        self.update(r, f, self.r0, self.f0)

        loopmax = min(self.memory, self.iteration)
        a = torch.empty(
            (loopmax, self.n_configs, 1), dtype=torch.float64, device=self.device
        )

        # ## The algorithm itself:
        q = -f.view(self.n_configs, -1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = self.rho[i] * (self.s[i] * q).sum(-1, keepdim=True)
            q -= a[i] * self.y[i]

        z = self.H0 * q

        for i in range(loopmax):
            b = self.rho[i] * (self.y[i] * z).sum(-1, keepdim=True)
            z += self.s[i] * (a[i] - b)

        p = -z.view(self.n_configs, self.n_atoms, 3)
        # broadcast rather than materialize a full-size boolean mask
        p = p * (~relaxed).view(-1, 1, 1)
        # ##

        dr = self.determine_step(p) if normalize_step else p.view(-1, 3)
        dr = dr * self.damping

        self.inputs[properties.R] += dr.to(
            device=self.inputs[properties.R].device,
            dtype=self.inputs[properties.R].dtype,
        )
        # the forces the cached reduction belongs to are stale now
        self._max_sq_force_per_config = None

        self.iteration += 1
        self.r0 = r
        self.f0 = f

    def determine_step(self, dr: torch.Tensor) -> torch.Tensor:
        """Scale the step down to ``maxstep``, each structure on its own.

        Every atom of a structure is scaled by the same factor, so the step still
        points along the eigendirection.
        """
        dr = dr.view(self.n_configs, self.n_atoms, 3)
        longest_step = dr.pow(2).sum(-1).sqrt().max(dim=1, keepdim=True).values
        # clamp instead of branching: structures below maxstep are scaled by 1
        scale = (self.maxstep / longest_step).clamp(max=1.0)
        return (dr * scale.unsqueeze(-1)).view(-1, 3)

    def update(
        self,
        r: torch.Tensor,
        f: torch.Tensor,
        r0: Optional[torch.Tensor],
        f0: Optional[torch.Tensor],
    ) -> None:
        """Append the latest position and gradient difference to the history."""
        if self.iteration > 0:
            s0 = (r - r0).view(self.n_configs, -1)
            self.s.append(s0)

            # we use the gradient, which is minus the force
            y0 = (f0 - f).view(self.n_configs, -1)
            self.y.append(y0)

            ys0 = (y0 * s0).sum(-1, keepdim=True)
            self.rho.append(torch.where(ys0 > 1e-8, 1.0 / ys0, torch.zeros_like(ys0)))

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

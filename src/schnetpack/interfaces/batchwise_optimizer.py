import torch
import pickle
import time
import logging
import numpy as np
from math import sqrt
from tqdm import tqdm
from os.path import isfile
from schnetpack import properties

from ase.optimize.optimize import Dynamics
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io import write
from ase import Atoms


__all__ = ["TorchLBFGS", "ASELBFGS"]


class TorchLBFGS(torch.optim.LBFGS):
    """
    LBFGS optimizer that allows for relaxation of multiple structures in parallel. The approximation of the inverse
    hessian is shared across the entire batch (all structures). Hence, it is recommended to use this optimizer
    preferably for batches of similar structures/compositions. In other cases, please utilize the ASELBFGS optimizer,
    which is particularly constructed for batches of different structures/compositions.
    """

    def __init__(
        self,
        model,
        model_inputs,
        fixed_atoms_mask,
        logging_function=None,
        lr: float = 1.0,
        energy_key: str = "energy",
        force_key: str = "forces",
        position_key: str = properties.R,
    ):
        """
        Args:
            model (schnetpack.model.AtomisticModel): ml force field model
            model_inputs: input batch containing all structures
            fixed_atoms_mask (list(bool)): list of booleans indicating to atoms with positions fixed in space.
            logging_function: function that logs the structure of the systems during the relaxation
            lr (float): learning rate (default: 1)
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            position_key (str): name of atomic positions in model (default="_positions")
        """

        self.model = model
        self.energy_key = energy_key
        self.force_key = force_key
        self.position_key = position_key
        self.fixed_atoms_mask = fixed_atoms_mask
        self.model_inputs = model_inputs
        self.logging_function = logging_function
        self.fmax = None

        R = self.model_inputs[self.position_key]
        R.requires_grad = True
        super().__init__(params=[R], lr=lr)

    def _gather_flat_grad(self):
        """override this function to allow for keeping atoms fixed during the relaxation"""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        flat_grad = torch.cat(views, 0)
        if self.fixed_atoms_mask is not None:
            flat_grad[self.fixed_atoms_mask] = 0.0
        self.flat_grad = flat_grad
        return flat_grad

    def closure(self):
        results = self.model(self.model_inputs)
        self.zero_grad()
        loss = results[self.energy_key].sum()
        loss.backward()
        return loss

    def log(self, forces=None):
        """log relaxation results such as max force in the system"""
        if forces is None:
            forces = self.flat_grad.view(-1, 3)
        if not self.converged():
            logging.info("NOT CONVERGED")
        logging.info(
            "max. atomic force: {} eV/Ang".format(
                torch.sqrt((forces**2).sum(axis=1).max())
            )
        )

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.flat_grad.view(-1, 3)
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def run(self, fmax, max_opt_steps):
        """run relaxation"""
        self.fmax = fmax

        # optimization
        for opt_step in tqdm(range(max_opt_steps)):
            self.step(self.closure)

            # log structure
            if self.logging_function is not None:
                self.logging_function(opt_step)

            # stop optimization if max force is smaller than threshold
            if self.converged():
                break
        self.log()

    def get_relaxed_structure(self):
        return self.model_inputs[self.position_key]


class DynamicsCustom(Dynamics):
    """Base-class for all MD and structure optimization classes."""

    def __init__(
        self,
        model,
        model_inputs,
        logfile,
        trajectory,
        fixed_atoms_mask=None,
        append_trajectory=False,
        master=None,
    ):
        super().__init__(None, logfile, trajectory, append_trajectory, master)
        self.model = model
        self.model.do_postprocessing = False
        self.fixed_atoms_mask = fixed_atoms_mask
        # convert ase object to model input
        self.model_inputs = model_inputs
        self.previous_pos = None
        self.model_results = None
        self.device = (
            torch.device("cuda")
            if model_inputs[properties.R].is_cuda
            else torch.device("cpu")
        )

    def _requires_calculation(self, properties):
        if self.model_results is None or self.previous_pos is None:
            return True
        for name in properties:
            if name not in self.model_results:
                return True
        return not torch.equal(self.previous_pos, self.model_inputs["_positions"])

    def _get_forces(self, inputs):
        if self._requires_calculation(properties=["forces", "energy"]):
            self.model_results = self.model(inputs)
            self.previous_pos = self.model_inputs["_positions"].clone()
        f = self.model_results["forces"]
        if self.fixed_atoms_mask is not None:
            f[self.fixed_atoms_mask] *= 0.0
        return f

    def _get_potential_energy(self, model_inputs):
        if self._requires_calculation(properties=["energy"]):
            self.model_results = self.model(model_inputs)
        return self.model_results["energy"]

    def irun(self):
        # compute initial structure and log the first step
        self._get_forces(self.model_inputs)

        # in the following we are assuming that index_m is sorted
        dev_from_sorted_index_m = (
            self.model_inputs[properties.idx_m].sort()[0]
            - self.model_inputs[properties.idx_m]
        )
        if abs(dev_from_sorted_index_m).sum() > 0.0:
            raise ValueError(
                "idx_m is assumed to be sorted, this is not the case here!"
            )

        # yield the first time to inspect before logging
        yield False

        if self.nsteps == 0:
            self.log()
            # self.call_observers()
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
            # self.log()
            # self.call_observers()

        # log last step
        self.log()

        # finally check if algorithm was converged
        yield self.converged()

    def run(self):
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        for converged in DynamicsCustom.irun(self):
            pass
        return converged


class OptimizerCustom(DynamicsCustom):
    """Base-class for all structure optimization classes."""

    # default maxstep for all optimizers
    defaults = {"maxstep": 0.2}

    def __init__(
        self,
        model,
        model_inputs,
        fixed_atoms_mask,
        restart,
        logfile,
        trajectory,
        master=None,
        append_trajectory=False,
        force_consistent=False,
    ):
        """Structure optimizer object.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: str
            Filename for restart file.  Default value is *None*.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        append_trajectory: boolean
            Appended to the trajectory file instead of overwriting it.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  If force_consistent=None, uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        DynamicsCustom.__init__(
            self,
            model=model,
            model_inputs=model_inputs,
            fixed_atoms_mask=fixed_atoms_mask,
            logfile=logfile,
            trajectory=trajectory,
            master=master,
            append_trajectory=append_trajectory,
        )

        self.force_consistent = force_consistent
        if self.force_consistent is None:
            self.set_force_consistent()

        self.restart = restart

        # initialize attribute
        self.fmax = None

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            barrier()

    def todict(self):
        description = {
            "type": "optimization",
            "optimizer": self.__class__.__name__,
        }
        return description

    def initialize(self):
        pass

    def irun(self, fmax=0.05, steps=None):
        """call Dynamics.irun and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return DynamicsCustom.irun(self)

    def run(self, fmax=0.05, steps=None):
        """call Dynamics.run and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return DynamicsCustom.run(self)

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self._get_forces(self.model_inputs)
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def log(self, forces=None):
        if forces is None:
            forces = self._get_forces(self.model_inputs)
        fmax = sqrt((forces**2).sum(axis=1).max())
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "fmax")
                msg = "%s  %4s %8s %12s\n" % args
                self.logfile.write(msg)

                if self.force_consistent:
                    msg = "*Force-consistent energies used in optimization.\n"
                    self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], fmax)
            msg = "%s:  %3d %02d:%02d:%02d %12.4f\n" % args
            self.logfile.write(msg)

            self.logfile.flush()

        if self.trajectory is not None:
            for struc_idx, _ in enumerate(self.model_inputs[properties.n_atoms]):
                R_m = self.model_inputs[properties.R][
                    self.model_inputs[properties.idx_m] == struc_idx
                ]
                Z_m = self.model_inputs[properties.Z][
                    self.model_inputs[properties.idx_m] == struc_idx
                ]
                at = Atoms(
                    positions=R_m.detach().cpu().numpy(),
                    numbers=Z_m.detach().cpu().numpy(),
                )
                at.pbc = (
                    self.model_inputs[properties.pbc][struc_idx].detach().cpu().numpy()
                )
                at.cell = (
                    torch.diag(self.model_inputs[properties.cell][struc_idx])
                    .detach()
                    .cpu()
                    .numpy()
                )
                # store in trajectory
                write(
                    self.trajectory + "_{}.xyz".format(struc_idx),
                    at,
                    format="extxyz",
                    append=False if self.nsteps == 0 else True,
                )

    def get_relaxed_strctures(self):
        ats = []
        for struc_idx, _ in enumerate(self.model_inputs[properties.n_atoms]):
            R_m = self.model_inputs[properties.R][
                self.model_inputs[properties.idx_m] == struc_idx
            ]
            Z_m = self.model_inputs[properties.Z][
                self.model_inputs[properties.idx_m] == struc_idx
            ]
            at = Atoms(
                positions=R_m.detach().cpu().numpy(), numbers=Z_m.detach().cpu().numpy()
            )
            at.pbc = self.model_inputs[properties.pbc][struc_idx].detach().cpu().numpy()
            at.cell = (
                torch.diag(self.model_inputs[properties.cell][struc_idx])
                .detach()
                .cpu()
                .numpy()
            )
            ats.append(at)
        return ats

    def get_relaxation_results(self):
        self.model.do_postprocessing = True
        # one more forward pass to get forces and energy of relaxed structure
        model_out = self.model(self.model_inputs)
        # prepare relaxation results for output parser
        relaxation_results = {"positions": self.model_inputs[properties.R]}
        relaxation_results.update(model_out)
        return relaxation_results

    def dump(self, data):
        if world.rank == 0 and self.restart is not None:
            with open(self.restart, "wb") as fd:
                pickle.dump(data, fd, protocol=2)

    def load(self):
        with open(self.restart, "rb") as fd:
            return pickle.load(fd)

    def set_force_consistent(self):
        """Automatically sets force_consistent to True if force_consistent
        energies are supported by calculator; else False."""
        try:
            self.atoms.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            self.force_consistent = False
        else:
            self.force_consistent = True


class ASELBFGS(OptimizerCustom):
    """Limited memory BFGS optimizer.

    A limited memory version of the bfgs algorithm. Unlike the bfgs algorithm
    used in bfgs.py, the inverse of Hessian matrix is updated.  The inverse
    Hessian is represented only as a diagonal matrix to save memory

    """

    def __init__(
        self,
        model,
        model_inputs,
        fixed_atoms_mask=None,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        memory=100,
        damping=1.0,
        alpha=70.0,
        use_line_search=False,
        master=None,
        force_consistent=None,
    ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store vectors for updating the inverse of
            Hessian matrix. If set, file with such a name will be searched
            and information stored will be used, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        maxstep: float
            How far is a single atom allowed to move. This is useful for DFT
            calculations where wavefunctions can be reused if steps are small.
            Default is 0.2 Angstrom.

        memory: int
            Number of steps to be stored. Default value is 100. Three numpy
            arrays of this length containing floats are stored.

        damping: float
            The calculated step is multiplied with this number before added to
            the positions.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """

        OptimizerCustom.__init__(
            self,
            model=model,
            model_inputs=model_inputs,
            fixed_atoms_mask=fixed_atoms_mask,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            master=master,
            force_consistent=force_consistent,
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
        self.trajectory = trajectory
        self.n_configs = None

    def initialize(self):
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

    def read(self):
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

    def step(self, f=None):
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it"""

        self.n_configs = self.model_inputs[properties.n_atoms].shape[0]

        if f is None:
            f = self._get_forces(self.model_inputs)

        r = self.model_inputs["_positions"]

        self.update(r, f, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = torch.empty(
            (
                loopmax,
                self.n_configs,
            ),
            dtype=torch.float64,
        ).to(self.device)

        # ## The algorithm itself:
        q = -f.view(self.n_configs, -1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * (s[i] * q).sum(-1)
            q -= torch.unsqueeze(a[i], -1) * y[i]

        z = H0 * q

        for i in range(loopmax):
            b = rho[i] * (y[i] * z).sum(-1)
            z += s[i] * torch.unsqueeze((a[i] - b), -1)

        self.p = -z.view((-1, 3))
        # ##

        g = -f
        if self.use_line_search is True:
            e = self.func(r)
            self.line_search(r, g, e)
            dr = (self.alpha_k * self.p).reshape(
                self.model_inputs["_n_atoms"].item(), -1
            )
        else:
            self.force_calls += 1
            self.function_calls += 1
            dr = self.determine_step(self.p) * self.damping
        self.model_inputs["_positions"] = self.model_inputs["_positions"] + dr

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

    def determine_step(self, dr):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        steplengths = (dr**2).sum(1) ** 0.5
        # check if any step in entire batch is greater than maxstep
        if torch.max(steplengths) >= self.maxstep:
            # rescale steps for each config separately
            for idx_m in range(self.n_configs):
                longest_step = torch.max(
                    steplengths[self.model_inputs[properties.idx_m] == idx_m]
                )
                if longest_step >= self.maxstep:
                    print("normalized integration step")
                    dr[self.model_inputs[properties.idx_m] == idx_m] *= (
                        self.maxstep / longest_step
                    )
        return dr

    def update(self, r, f, r0, f0):
        """Update everything that is kept in memory

        This function is mostly here to allow for replay_trajectory.
        """
        if self.iteration > 0:
            s0 = r.view(self.n_configs, -1) - r0.view(self.n_configs, -1)
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0.view(self.n_configs, -1) - f.view(self.n_configs, -1)
            self.y.append(y0)

            rho0 = 1.0 / ((s0 * y0).sum(-1) + 1e-16)
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

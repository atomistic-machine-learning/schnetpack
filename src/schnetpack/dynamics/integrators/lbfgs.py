"""Limited-memory BFGS as a step rule for relaxation."""

import dataclasses

import torch

from schnetpack.dynamics.integrators.base import Integrator

__all__ = ["LBFGS", "LBFGSState"]


@dataclasses.dataclass
class LBFGSState:
    """
    The history :class:`LBFGS` carries from one step to the next.

    Attributes:
        n_structures: structures in the batch
        n_atoms: atoms per structure
        s: position differences of the last steps, each
            ``(n_structures, 3 n_atoms)``
        y: gradient differences of the last steps, shaped like ``s``
        rho: ``1 / (y . s)`` per structure, ``(n_structures, 1)`` each
        x0: positions of the previous step
        f0: forces of the previous step
        iteration: steps taken
    """

    n_structures: int
    n_atoms: int
    s: list = dataclasses.field(default_factory=list)
    y: list = dataclasses.field(default_factory=list)
    rho: list = dataclasses.field(default_factory=list)
    x0: torch.Tensor | None = None
    f0: torch.Tensor | None = None
    iteration: int = 0


class LBFGS(Integrator):
    """
    Limited-memory BFGS, one inverse-Hessian approximation per structure.

    An adaptation of ``ase.optimize.LBFGS`` to a batch: the two-loop
    recursion, the curvature pairs and the step length are all per
    structure, so structures of different compositions relax together
    exactly as they would alone.

    A step rule for a :class:`~schnetpack.dynamics.relax.Relaxer`: it follows
    the field's drift as forces (eV/Angstrom on positions in Angstrom) and
    ignores ``t`` and ``dt`` — the step length is set by ``maxstep`` and
    ``damping``. It reads the structure layout off the field
    (``requires_structure``); holding converged structures still is the
    relaxer's job. All structures of a batch must have the same number of
    atoms.

    The history lives in an :class:`LBFGSState`, one per run. A constraint
    that moves atoms between steps (e.g. noise injection) makes the history
    describe a path the structures did not take; pure overwrites of fixed
    atoms are harmless, since their forces are zeroed by the relaxer.
    """

    requires_structure = True

    def __init__(
        self,
        maxstep: float = 0.2,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            maxstep: how far a single atom may move in one step, in Angstrom.
                Each structure is rescaled on its own.
            memory: steps of history kept for the two-loop recursion
            damping: the computed step is multiplied by this before it is
                taken
            alpha: initial guess for the curvature of the energy surface. The
                conservative default of 70.0 emulates BFGS; a lower value may
                converge in fewer steps at the cost of stability.
            device: device the recursion runs on. It is bound by kernel
                launches rather than arithmetic — measured 5-7x slower on
                cuda than on cpu for batches up to 256 structures of 1000
                atoms — so the default is cpu wherever the model runs. Worth
                re-measuring before overriding for much larger batches.
        """
        if maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for the maximum step "
                f"size: {maxstep:.1f} Angstrom"
            )
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        # initial inverse Hessian, 1/70 to emulate BFGS; never changed
        self.H0 = 1.0 / alpha
        self.device = torch.device(device)

    def init_state(self, dynamics, x) -> LBFGSState:
        n_atoms = dynamics.n_atoms
        if not bool((n_atoms == n_atoms[0]).all()):
            raise ValueError(
                "LBFGS requires all structures in the batch to have the same "
                f"number of atoms, got {n_atoms.tolist()}"
            )
        return LBFGSState(n_structures=n_atoms.shape[0], n_atoms=int(n_atoms[0]))

    def step(self, dynamics, x, t, dt, state):
        n_structures, n_atoms = state.n_structures, state.n_atoms
        f = dynamics.drift(x, t).to(device=self.device, dtype=torch.float64)
        r = x.to(device=self.device, dtype=torch.float64)

        state = self._update(state, r, f)
        loopmax = min(self.memory, state.iteration)
        a = torch.empty(
            (loopmax, n_structures, 1), dtype=torch.float64, device=self.device
        )

        q = -f.view(n_structures, -1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = state.rho[i] * (state.s[i] * q).sum(-1, keepdim=True)
            q -= a[i] * state.y[i]

        z = self.H0 * q
        for i in range(loopmax):
            b = state.rho[i] * (state.y[i] * z).sum(-1, keepdim=True)
            z += state.s[i] * (a[i] - b)

        p = -z.view(n_structures, n_atoms, 3)
        dr = self._determine_step(p) * self.damping

        state.iteration += 1
        state.x0 = r
        state.f0 = f
        return x + dr.to(device=x.device, dtype=x.dtype), state

    def _determine_step(self, dr: torch.Tensor) -> torch.Tensor:
        """
        Scale the step down to ``maxstep``, each structure on its own.

        Every atom of a structure is scaled by the same factor, so the step
        still points along the eigendirection.
        """
        longest = dr.pow(2).sum(-1).sqrt().max(dim=1, keepdim=True).values
        # clamp instead of branching: structures below maxstep are scaled by 1
        scale = (self.maxstep / longest).clamp(max=1.0)
        return (dr * scale.unsqueeze(-1)).view(-1, 3)

    def _update(self, state: LBFGSState, r, f) -> LBFGSState:
        """Append the latest position and gradient difference to the history."""
        if state.iteration > 0:
            s0 = (r - state.x0).view(state.n_structures, -1)
            state.s.append(s0)
            # the gradient is minus the force
            y0 = (state.f0 - f).view(state.n_structures, -1)
            state.y.append(y0)
            ys0 = (y0 * s0).sum(-1, keepdim=True)
            state.rho.append(torch.where(ys0 > 1e-8, 1.0 / ys0, torch.zeros_like(ys0)))

        if state.iteration > self.memory:
            state.s.pop(0)
            state.y.pop(0)
            state.rho.pop(0)
        return state

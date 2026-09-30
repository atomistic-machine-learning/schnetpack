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
        idx_m: ``(n_total_atoms,)`` structure of every atom, on the
            recursion's device
        s: position differences of the last steps, each
            ``(n_total_atoms, 3)``
        y: gradient differences of the last steps, shaped like ``s``
        rho: ``1 / (y . s)`` per structure, ``(n_structures,)`` each
        x0: positions of the previous step
        f0: forces of the previous step
        iteration: steps taken
    """

    n_structures: int
    idx_m: torch.Tensor
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
    structure, so structures of different sizes and compositions relax
    together exactly as they would alone. The recursion runs on the batch's
    own layout, the atoms of every structure end to end, and reduces per
    structure over ``idx_m``.

    A step rule for a :class:`~schnetpack.dynamics.relax.Relaxer`: it follows
    the field's drift as forces (eV/Angstrom on positions in Angstrom) and
    ignores ``t`` and ``dt`` — the step length is set by ``maxstep`` and
    ``damping``. It reads the structure layout off the field
    (``requires_structure``); holding converged structures still is the
    relaxer's job.

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
                launches rather than arithmetic — measured 2-3x slower on
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
        self.H0 = 1.0 / alpha  # initial inverse Hessian
        self.device = torch.device(device)

    def init_state(self, dynamics, x) -> LBFGSState:
        # the layout is fixed for a run: read it once, not on every step
        return LBFGSState(
            n_structures=dynamics.n_atoms.shape[0],
            idx_m=dynamics.idx_m.to(self.device),
        )

    def step(self, dynamics, x, t, dt, state):
        f = dynamics.drift(x, t).to(device=self.device, dtype=torch.float64)
        r = x.to(device=self.device, dtype=torch.float64)

        state = self._update(state, r, f)
        loopmax = min(self.memory, state.iteration)
        a = [None] * loopmax
        idx_m = state.idx_m

        q = -f
        for i in range(loopmax - 1, -1, -1):
            a[i] = state.rho[i] * self._dot(state, state.s[i], q)
            q.addcmul_(
                a[i].index_select(0, idx_m).unsqueeze(-1), state.y[i], value=-1.0
            )

        z = self.H0 * q
        for i in range(loopmax):
            b = state.rho[i] * self._dot(state, state.y[i], z)
            z.addcmul_((a[i] - b).index_select(0, idx_m).unsqueeze(-1), state.s[i])

        dr = self._determine_step(state, -z) * self.damping

        state.iteration += 1
        state.x0 = r
        state.f0 = f
        return x + dr.to(device=x.device, dtype=x.dtype), state

    @staticmethod
    def _dot(state: LBFGSState, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The dot product of ``u`` and ``v`` per structure, ``(n_structures,)``."""
        # a matrix-vector product sums over xyz: a reduction over a last axis
        # of 3 is several times slower on cpu
        per_atom = (u * v) @ u.new_ones(3)
        return u.new_zeros(state.n_structures).index_add_(0, state.idx_m, per_atom)

    def _determine_step(self, state: LBFGSState, dr: torch.Tensor) -> torch.Tensor:
        """
        Scale the step down to ``maxstep``, each structure on its own.

        Every atom of a structure is scaled by the same factor, so the step
        still points along the eigendirection.
        """
        longest = torch.zeros(
            state.n_structures, dtype=dr.dtype, device=dr.device
        ).scatter_reduce(0, state.idx_m, dr.norm(dim=-1), "amax", include_self=True)
        # clamp instead of branching: structures below maxstep are scaled by 1
        scale = (self.maxstep / longest).clamp(max=1.0)
        return dr * scale[state.idx_m, None]

    def _update(self, state: LBFGSState, r, f) -> LBFGSState:
        """Append the latest position and gradient difference to the history."""
        if state.iteration > 0:
            s0 = r - state.x0
            state.s.append(s0)
            # the gradient is minus the force
            y0 = state.f0 - f
            state.y.append(y0)
            ys0 = self._dot(state, y0, s0)
            state.rho.append(torch.where(ys0 > 1e-8, 1.0 / ys0, torch.zeros_like(ys0)))

        if state.iteration > self.memory:
            state.s.pop(0)
            state.y.pop(0)
            state.rho.pop(0)
        return state

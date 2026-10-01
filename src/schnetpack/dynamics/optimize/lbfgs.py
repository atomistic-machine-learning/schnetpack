"""Limited-memory BFGS relaxation, one inverse Hessian per structure."""

import dataclasses
from collections.abc import Sequence

import torch

from schnetpack import properties
from schnetpack.dynamics.optimize.optimizer import Optimizer
from schnetpack.generative.priors import Prior

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


class LBFGS(Optimizer):
    """
    Limited-memory BFGS, one inverse-Hessian approximation per structure.

    An adaptation of ``ase.optimize.LBFGS`` to a batch: the two-loop
    recursion, the curvature pairs and the step length are all per
    structure, so structures of different sizes and compositions relax
    together exactly as they would alone. The recursion runs on the batch's
    own layout, the atoms of every structure end to end, and reduces per
    structure over ``idx_m``. The step length is set by ``maxstep`` and
    ``damping``; the loop, the stop test and the holding of converged
    structures and fixed atoms are :class:`~schnetpack.dynamics.optimize.Optimizer`'s.

    The starting curvature ``alpha`` depends on the force's kind: 70
    eV/Angstrom^2 for a physical force, as ase's BFGS; 2 for a pseudo-force
    F = 2 (x0 - x), whose first step x + F/2 is then exactly GPFF's jump to
    the x0-estimate (when ``maxstep`` allows it).

    The history lives in an :class:`LBFGSState`, one per run. A state
    constraint that moves atoms between steps makes the history describe a
    path the structures did not take. Pure overwrites of fixed atoms are
    harmless, since their forces are zeroed by the loop.
    """

    def __init__(
        self,
        calculator,
        fmax: float | None = 0.05,
        maxstep: float = 0.2,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float | None = None,
        device: str | torch.device = "cpu",
        prior: Prior | None = None,
        constraints: Sequence = (),
        key: str = properties.R,
    ):
        """
        Args:
            calculator: see :class:`~schnetpack.dynamics.optimize.Optimizer`
            fmax: stop criterion on the largest force, in the force's unit
                (eV/Angstrom, or Angstrom for a pseudo-force); None runs all
                ``n_steps``
            maxstep: how far a single atom may move in one step, in Angstrom.
                Each structure is rescaled on its own.
            memory: steps of history kept for the two-loop recursion
            damping: the computed step is multiplied by this before it is
                taken
            alpha: initial guess for the curvature of the surface (default:
                70.0 for a physical force, which emulates BFGS; 2.0 for a
                pseudo-force). A lower value may converge in fewer steps at
                the cost of stability.
            device: device the recursion runs on. It is bound by kernel
                launches rather than arithmetic — measured 2-3x slower on
                cuda than on cpu for batches up to 256 structures of 1000
                atoms — so the default is cpu wherever the model runs. Worth
                re-measuring before overriding for much larger batches.
            prior: see :class:`~schnetpack.dynamics.optimize.Optimizer`
            constraints: see :class:`~schnetpack.dynamics.optimize.Optimizer`
            key: see :class:`~schnetpack.dynamics.optimize.Optimizer`
        """
        if maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for the maximum step "
                f"size: {maxstep:.1f} Angstrom"
            )
        super().__init__(
            calculator, prior=prior, constraints=constraints, key=key, fmax=fmax
        )
        if alpha is None:
            alpha = 70.0 if self.calculator.physical else 2.0
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.H0 = 1.0 / alpha  # initial inverse Hessian
        self.device = torch.device(device)

    def init_state(self, batch, x) -> LBFGSState:
        # the layout is fixed for a run: read it once, not on every step
        return LBFGSState(
            n_structures=batch[properties.n_atoms].shape[0],
            idx_m=batch[properties.idx_m].to(self.device),
        )

    def step(self, batch, x, forces, state):
        f = forces.to(device=self.device, dtype=torch.float64)
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

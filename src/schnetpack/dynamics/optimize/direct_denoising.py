"""
GPFF's direct denoising: a relaxation on a pseudo-force, not a time-stepped
sampler.

The loop repeats "jump to the model's x0-estimate"; there is no time grid and no reverse SDE. The jump x <- x + F/2 is the Newton step on the
pseudo-energy ||x - x0||^2. Details: ``docs_new/sampling.md`` §5.
"""

from collections.abc import Sequence

from schnetpack import properties
from schnetpack.dynamics.calculator import Calculator, ForceCalculator
from schnetpack.dynamics.optimize.optimizer import Optimizer
from schnetpack.generative.priors import Prior

__all__ = ["DirectDenoising"]


class DirectDenoising(Optimizer):
    """
    GPFF's direct denoising: repeat "jump to the x0-estimate".

    Each of the ``n_steps`` iterations does x <- x + F/2, F the pseudo-force
    2 (x0 - x). The calculator must be a pseudo-force
    :class:`~schnetpack.dynamics.calculator.ForceCalculator`: it shows the
    model the path time 0 and returns F in Angstrom. The model must therefore ignore its time input.
    Time-conditioned models belong in
    :class:`~schnetpack.dynamics.sample.Sampler`.

    There is no process to derive a starting distribution from: :meth:`sample`
    needs an explicit ``prior``, e.g. the training process's
    ``sampling_prior()``. Guidance on the calculator enters the pseudo-force,
    and with it the jump, as w F with w in Angstrom^2/eV (see
    :class:`~schnetpack.dynamics.calculator.ForceCalculator`).
    """

    def __init__(
        self,
        calculator,
        prior: Prior | None = None,
        fmax: float | None = None,
        hooks: Sequence = (),
        key: str = properties.R,
    ):
        """
        Args:
            calculator: a pseudo-force
                :class:`~schnetpack.dynamics.calculator.ForceCalculator`, or a
                bare callable batch -> outputs, taken as one in Angstrom
            prior: starting distribution :meth:`sample` draws from
            fmax: stop criterion on the largest pseudo-force, in Angstrom;
                None runs all ``n_steps``
            hooks: run around every step, in order
            key: batch key this driver moves
        """
        if not isinstance(calculator, Calculator):
            calculator = ForceCalculator(calculator, kind="pseudo")
        if isinstance(calculator, ForceCalculator) and calculator.physical:
            raise TypeError(
                f"{type(self).__name__} jumps along a pseudo-force; got a "
                "physical ForceCalculator"
            )
        super().__init__(
            calculator,
            prior=prior,
            hooks=hooks,
            key=key,
            fmax=fmax,
        )

    def step(self, batch, x, forces, state):
        return x + 0.5 * forces, state

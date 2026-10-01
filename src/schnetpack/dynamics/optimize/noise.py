"""
Noise schedules of the time-free family: how much Gaussian noise an
:class:`~schnetpack.dynamics.optimize.Optimizer` loop injects before each step.

A schedule is a callable ``(step, n_steps) -> scale``; the loop adds
``scale * z``, z ~ N(0, I), to the moved atoms before the step's force is
evaluated. The scale is a length, in Angstrom like the batch.
"""

import abc

__all__ = ["NoiseSchedule", "ConstantNoise", "AnnealedNoise"]


class NoiseSchedule(abc.ABC):
    """Base class of the noise schedules."""

    @abc.abstractmethod
    def __call__(self, step: int, n_steps: int) -> float:
        """
        The noise scale injected before step ``step`` of ``n_steps``.

        Args:
            step: index of the step about to run, 0-based
            n_steps: the run's step limit

        Returns:
            The scale, in Angstrom; 0 injects nothing.
        """
        raise NotImplementedError


class ConstantNoise(NoiseSchedule):
    """The same scale before every step, e.g. Langevin's sqrt(2 eps kT)."""

    def __init__(self, scale: float):
        """
        Args:
            scale: noise scale, in Angstrom
        """
        self.scale = scale

    def __call__(self, step, n_steps):
        return self.scale


class AnnealedNoise(NoiseSchedule):
    """
    GPFF's decaying noise, lambda (1 - k/N) before step k = 1..N. The last
    step injects nothing.
    """

    def __init__(self, stochastic_lambda: float = 1.0):
        """
        Args:
            stochastic_lambda: scale of the first injection's decay, in
                Angstrom
        """
        self.stochastic_lambda = stochastic_lambda

    def __call__(self, step, n_steps):
        return max(self.stochastic_lambda * (1.0 - (step + 1) / n_steps), 0.0)

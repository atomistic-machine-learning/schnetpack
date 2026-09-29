"""
What a relaxation adds to the frames it reports, and its progress log.

The observer seam itself — :class:`~schnetpack.dynamics.observers.Observer`,
the trajectory and in-memory recorders — is shared by every driver and lives
in :mod:`schnetpack.dynamics.observers`.
"""

import dataclasses
import sys
import time
from typing import TextIO

import torch

from schnetpack.dynamics.observers import Frame, Interval, Observer

__all__ = ["RelaxationFrame", "LogWriter"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RelaxationFrame(Frame):
    """
    A frame of a relaxation.

    Attributes:
        energy: ``(n_structures,)``, in eV
        forces: ``(n_total_atoms, 3)``, in eV/Angstrom
        max_force_per_config: ``(n_structures,)``, the largest force on any
            free atom of each structure, in eV/Angstrom — what the
            convergence criterion is applied to
        fmax: the force criterion the run is held to
    """

    energy: torch.Tensor
    forces: torch.Tensor
    max_force_per_config: torch.Tensor
    fmax: float

    trajectory_fields = (
        ("energy", "energy"),
        ("converged", "converged"),
        ("forces", "forces"),
    )

    @property
    def converged(self) -> torch.Tensor:
        """``(n_structures,)``, which structures already meet the criterion."""
        return self.max_force_per_config < self.fmax

    @property
    def max_force(self) -> float:
        """The largest force anywhere in the batch, in eV/Angstrom."""
        return self.max_force_per_config.max().item()


class LogWriter(Observer):
    """
    Writes one line of progress per recorded step of a relaxation.

    A path is opened in append mode and closed by :meth:`on_end`; stdout and
    a file object passed in are left open, since this observer did not open
    them.
    """

    def __init__(self, logfile, interval: int = 1):
        """
        Args:
            logfile: a path, ``"-"`` for stdout, or a file object open for
                writing
            interval: how often to write a line, see
                :class:`~schnetpack.dynamics.observers.Interval`
        """
        self.logfile = logfile
        self.interval = Interval(interval)
        self.file: TextIO | None = None
        self._owned = False
        self._name = "Relaxation"

    def on_start(self, dynamics, batch) -> None:
        self.on_end()
        if self.logfile == "-":
            self.file = sys.stdout
        elif isinstance(self.logfile, str):
            self.file = open(self.logfile, "a", encoding="utf-8")
            self._owned = True
        else:
            self.file = self.logfile
        # named after the step rule, the way ase labels its optimizers' logs
        integrator = getattr(dynamics, "integrator", None)
        self._name = type(integrator if integrator is not None else dynamics).__name__

    def wants(self, step: int, final: bool) -> bool:
        return self.interval.due(step, final)

    def on_frame(self, frame: RelaxationFrame) -> None:
        if self.file is None:
            return
        if frame.step == 0:
            pad = " " * len(self._name)
            self.file.write(f"{pad}  {'Step':>4s} {'Time':>8s} {'fmax':>12s}\n")
        clock = time.localtime()
        self.file.write(
            f"{self._name}:  {frame.step:3d} "
            f"{clock[3]:02d}:{clock[4]:02d}:{clock[5]:02d} {frame.max_force:12.4f}\n"
        )
        self.file.flush()

    def on_end(self) -> None:
        if self._owned and self.file is not None:
            self.file.close()
        self.file = None
        self._owned = False

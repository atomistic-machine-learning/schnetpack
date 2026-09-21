"""What a batch-wise relaxation reports while it runs.

A relaxation drives structures downhill; where the progress log, the trajectory or
anything else built from the intermediate states goes is a separate question.
:class:`RelaxationObserver` is the seam between the two. The optimizer builds one
:class:`RelaxationFrame` per recorded step and hands it to every observer it was given;
observers decide what to do with it and own whatever files they open.

This mirrors ``simulator_hooks`` on the MD side (see
:class:`~schnetpack.md.Simulator`), so the two drivers of this package report in the same
way.

Three observers ship here:

``LogWriter``
    the text progress log, one line per step.
``TrajectoryRecorder``
    the HDF5 trajectory, via :class:`~schnetpack.relax.BatchwiseTrajectoryWriter`.
``FrameCollector``
    keeps the frames in memory. For tests and for notebooks that want the path of a
    relaxation without a file on disk.

Building a frame costs a transfer off the device, so the optimizer asks
:meth:`RelaxationObserver.wants` first and only builds one when some observer says yes.
That is why the step interval lives here, in :class:`Interval`, rather than in the run
loop.
"""

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, TextIO

import torch

from schnetpack import properties
from schnetpack.relax.batchwise_trajectory import BatchwiseTrajectoryWriter

if TYPE_CHECKING:
    from schnetpack.relax.batchwise_optimization import BatchwiseOptimizer

__all__ = [
    "RelaxationFrame",
    "RelaxationObserver",
    "Interval",
    "LogWriter",
    "TrajectoryRecorder",
    "FrameCollector",
]


@dataclass(frozen=True)
class RelaxationFrame:
    """One recorded state of the batch.

    The tensors are the optimizer's live ones, at its precision and on its device --
    an observer that keeps a frame beyond the call has to clone them.

    Attributes:
        step: optimizer steps taken when this frame was recorded.
        final: this is the last frame of the run, because the batch converged or the
            step limit was reached.
        positions: ``(n_structures * n_atoms, 3)``, flat, as the optimizer holds them.
        cell: ``(n_structures, 3, 3)``.
        energy: ``(n_structures,)``, in eV.
        forces: ``(n_structures * n_atoms, 3)``, in eV/Angstrom.
        max_force_per_config: ``(n_structures,)``, the largest force on any free atom of
            each structure, in eV/Angstrom. Fixed atoms are excluded, so this is what
            the convergence criterion is applied to.
        fmax: the force criterion the run is being held to.
    """

    step: int
    final: bool
    positions: torch.Tensor
    cell: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    max_force_per_config: torch.Tensor
    fmax: float

    @property
    def converged(self) -> torch.Tensor:
        """``(n_structures,)``, which structures already meet the criterion."""
        return self.max_force_per_config < self.fmax

    @property
    def max_force(self) -> float:
        """The largest force anywhere in the batch, in eV/Angstrom."""
        return self.max_force_per_config.max().item()


class RelaxationObserver(ABC):
    """Something that watches a relaxation go by.

    Only :meth:`on_frame` has to be implemented. :meth:`on_start` and :meth:`on_end`
    default to doing nothing, and :meth:`wants` to accepting every frame.
    """

    def on_start(self, optimizer: "BatchwiseOptimizer") -> None:
        """Called once, from ``irun``, before the first frame.

        ``fmax`` is settled by this point, and the batch is still in its initial state.
        """

    def wants(self, step: int, final: bool) -> bool:
        """Is a frame for this step worth building?

        Called before the frame exists, so an observer that only records occasionally
        does not make the run pay for the frames it drops.
        """
        return True

    @abstractmethod
    def on_frame(self, frame: RelaxationFrame) -> None:
        """Called for every frame at least one observer asked for.

        An observer that returned ``False`` from :meth:`wants` for this step is not
        called, even when another observer wanted the frame.
        """

    def on_end(self) -> None:
        """Release whatever this observer opened. Must tolerate being called twice."""


class Interval:
    """How often a step is recorded.

    ``0`` records only the first and last frame, ``1`` every step, ``n`` every nth. The
    first and last frame are recorded whatever the interval.
    """

    def __init__(self, interval: int = 1):
        self.interval = interval

    def due(self, step: int, final: bool) -> bool:
        if final or step == 0:
            return True
        return self.interval > 0 and step % self.interval == 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.interval})"


class LogWriter(RelaxationObserver):
    """Writes one line of progress per recorded step.

    Args:
        logfile: a path, ``"-"`` for stdout, or a file object already open for writing.
            A path is opened in append mode and closed by :meth:`on_end`; stdout and a
            file object passed in are left open, since this observer did not open them.
        interval: how often to write a line, see :class:`Interval`.
    """

    def __init__(self, logfile, interval: int = 1):
        self.interval = Interval(interval)
        self._owned = False
        if logfile == "-":
            self.file: Optional[TextIO] = sys.stdout
        elif isinstance(logfile, str):
            self.file = open(logfile, "a", encoding="utf-8")
            self._owned = True
        else:
            self.file = logfile
        self._name = "Relaxation"

    def on_start(self, optimizer: "BatchwiseOptimizer") -> None:
        # the optimizer names the run, the way ase's optimizers label their log
        self._name = type(optimizer).__name__

    def wants(self, step: int, final: bool) -> bool:
        return self.interval.due(step, final)

    def on_frame(self, frame: RelaxationFrame) -> None:
        if self.file is None:
            return
        if frame.step == 0:
            header = (" " * len(self._name), "Step", "Time", "fmax")
            self.file.write("%s  %4s %8s %12s\n" % header)

        clock = time.localtime()
        line = (self._name, frame.step, clock[3], clock[4], clock[5], frame.max_force)
        self.file.write("%s:  %3d %02d:%02d:%02d %12.4f\n" % line)
        self.file.flush()

    def on_end(self) -> None:
        if self._owned and self.file is not None:
            self.file.close()
        self.file = None


class TrajectoryRecorder(RelaxationObserver):
    """Writes the recorded frames to a single HDF5 trajectory.

    The file is created in :meth:`on_start`, so its shape and its metadata are settled
    before the first frame rather than on whichever step happens to be recorded first.

    Args:
        filename: path of the HDF5 file to write.
        interval: how often to record a frame, see :class:`Interval`. Defaults to ``0``,
            i.e. the endpoints only -- a trajectory is much larger than a log.
        store_forces: also store the forces of every frame. Doubles the file size.
        writer_kwargs: passed on to
            :class:`~schnetpack.relax.BatchwiseTrajectoryWriter`, e.g. ``precision`` or
            ``buffer_size``.
    """

    def __init__(
        self,
        filename: str,
        interval: int = 0,
        store_forces: bool = False,
        **writer_kwargs,
    ):
        self.filename = filename
        self.interval = Interval(interval)
        self.store_forces = store_forces
        self.writer_kwargs = writer_kwargs
        self.writer: Optional[BatchwiseTrajectoryWriter] = None

    def on_start(self, optimizer: "BatchwiseOptimizer") -> None:
        self.writer = BatchwiseTrajectoryWriter(
            self.filename,
            atomic_numbers=optimizer.inputs[properties.Z],
            pbc=optimizer.inputs[properties.pbc],
            store_forces=self.store_forces,
            attrs={"optimizer": type(optimizer).__name__, "fmax": optimizer.fmax},
            **self.writer_kwargs,
        )

    def wants(self, step: int, final: bool) -> bool:
        return self.interval.due(step, final)

    def on_frame(self, frame: RelaxationFrame) -> None:
        if self.writer is None:
            raise RuntimeError(
                "TrajectoryRecorder.on_frame was called before on_start; the "
                "trajectory file has not been created yet"
            )
        self.writer.write(
            step=frame.step,
            positions=frame.positions,
            cell=frame.cell,
            energy=frame.energy,
            forces=frame.forces if self.store_forces else None,
            converged=frame.converged,
        )

    def on_end(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class FrameCollector(RelaxationObserver):
    """Keeps the recorded frames in memory, in :attr:`frames`.

    The optimizer hands out its live tensors, so the frames are cloned onto the cpu as
    they arrive -- otherwise every collected frame would show the final positions.

    Args:
        interval: how often to keep a frame, see :class:`Interval`.
    """

    def __init__(self, interval: int = 1):
        self.interval = Interval(interval)
        self.frames: List[RelaxationFrame] = []

    def wants(self, step: int, final: bool) -> bool:
        return self.interval.due(step, final)

    def on_frame(self, frame: RelaxationFrame) -> None:
        detached = {
            name: value.detach().cpu().clone()
            for name, value in vars(frame).items()
            if isinstance(value, torch.Tensor)
        }
        self.frames.append(
            RelaxationFrame(
                step=frame.step, final=frame.final, fmax=frame.fmax, **detached
            )
        )

    @property
    def steps(self) -> List[int]:
        """The optimizer step each collected frame belongs to."""
        return [frame.step for frame in self.frames]

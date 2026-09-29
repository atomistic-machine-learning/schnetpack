"""
What a :class:`~schnetpack.dynamics.base.Dynamics` loop reports while it runs.

A driver moves structures; where the progress log, the trajectory or anything
else built from the intermediate states goes is a separate question.
:class:`Observer` is the seam between the two. The driver builds a
:class:`Frame` per recorded step and hands it to every observer that asked
for it; observers decide what to do with it and own whatever files they
open. Every driver reports the same way: a sampler's reverse-diffusion path
and a relaxer's path through the energy landscape are recorded by the same
observers, each driver adding what it knows to the frame
(:class:`SamplingFrame`: the time;
:class:`~schnetpack.dynamics.relax.observers.RelaxationFrame`: energies,
forces and convergence).

Observers are called after the constraints of a step, so a frame always
shows the state the next step starts from.

Building a frame costs a transfer off the device, so the driver asks
:meth:`Observer.wants` first and only builds one when some observer says
yes. That is why the step interval lives here, in :class:`Interval`, rather
than in the driver's loop.

Two observers ship here: :class:`FrameCollector` keeps the frames in memory
(for tests and notebooks that want a path without a file), and
:class:`TrajectoryRecorder` writes them to HDF5 via
:class:`~schnetpack.dynamics.trajectory.TrajectoryWriter`.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from schnetpack import properties
from schnetpack.dynamics.trajectory import TrajectoryWriter

if TYPE_CHECKING:
    from schnetpack.dynamics.base import Dynamics

__all__ = [
    "Frame",
    "SamplingFrame",
    "Observer",
    "Interval",
    "FrameCollector",
    "TrajectoryRecorder",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Frame:
    """
    One recorded state of a run.

    The tensors are the driver's live ones, at its precision and on its
    device — an observer that keeps a frame beyond the call has to clone
    them (:class:`FrameCollector` does).

    Attributes:
        step: steps taken when this frame was recorded
        final: this is the last frame of the run
        positions: the moved key, ``(n_total_atoms, ...)``, flat as the
            driver holds it
        batch: the whole batch the positions belong to
        extras: anything else a driver reports, by name
    """

    step: int
    final: bool
    positions: torch.Tensor
    batch: Mapping[str, Any]
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)

    #: per-frame quantities a trajectory stores alongside the positions,
    #: ``(dataset name, attribute)``; subclasses extend it
    trajectory_fields = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplingFrame(Frame):
    """
    A frame of a reverse-process run.

    Attributes:
        t: path time of the iterate, 0-dim
    """

    t: torch.Tensor

    trajectory_fields = (("t", "t"),)


class Observer(ABC):
    """
    Something that watches a run go by.

    Only :meth:`on_frame` has to be implemented. :meth:`on_start` and
    :meth:`on_end` default to doing nothing, and :meth:`wants` to accepting
    every frame.
    """

    def on_start(self, dynamics: "Dynamics", batch: Mapping[str, Any]) -> None:
        """
        Called once per run, before the first frame, with the starting batch.

        The driver's run parameters (a relaxer's ``fmax``) are settled by now.
        """
        return None

    def wants(self, step: int, final: bool) -> bool:
        """
        Is a frame for this step worth building?

        Called before the frame exists, so an observer that only records
        occasionally does not make the run pay for the frames it drops.
        """
        return True

    @abstractmethod
    def on_frame(self, frame: Frame) -> None:
        """
        Called for every frame this observer asked for.
        """

    def on_end(self) -> None:
        """Release whatever this observer opened. Must tolerate being called twice."""
        return None


class Interval:
    """
    How often a step is recorded.

    ``0`` records only the first and last frame, ``1`` every step, ``n`` every
    nth. The first and last frame are recorded whatever the interval.
    """

    def __init__(self, interval: int = 1):
        self.interval = interval

    def due(self, step: int, final: bool) -> bool:
        if final or step == 0:
            return True
        return self.interval > 0 and step % self.interval == 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.interval})"


def _detached(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


class FrameCollector(Observer):
    """
    Keeps the recorded frames in memory, in :attr:`frames`.

    Drivers hand out their live tensors, so the tensors of a frame are cloned
    onto the cpu as they arrive. The batch is kept as a shallow copy: drivers
    build a new batch per step rather than editing one in place.
    """

    def __init__(self, interval: int = 1):
        """
        Args:
            interval: how often to keep a frame, see :class:`Interval`
        """
        self.interval = Interval(interval)
        self.frames: list[Frame] = []

    def on_start(self, dynamics, batch) -> None:
        self.frames = []

    def wants(self, step: int, final: bool) -> bool:
        return self.interval.due(step, final)

    def on_frame(self, frame: Frame) -> None:
        changes = {
            f.name: _detached(getattr(frame, f.name))
            for f in dataclasses.fields(frame)
            if torch.is_tensor(getattr(frame, f.name))
        }
        changes["batch"] = dict(frame.batch)
        changes["extras"] = {k: _detached(v) for k, v in frame.extras.items()}
        self.frames.append(dataclasses.replace(frame, **changes))

    @property
    def steps(self) -> list[int]:
        """The step each collected frame belongs to."""
        return [frame.step for frame in self.frames]


class TrajectoryRecorder(Observer):
    """
    Writes the recorded frames to a single HDF5 trajectory.

    The file is created in :meth:`on_start`, so its structure layout and its
    metadata are settled before the first frame. What is stored per frame
    besides positions and cell follows the frame's type — the time of a
    sampling run, the energies and convergence flags of a relaxation — see
    :class:`~schnetpack.dynamics.trajectory.TrajectoryWriter`.
    """

    def __init__(
        self,
        filename: str,
        interval: int = 0,
        store_forces: bool = False,
        **writer_kwargs,
    ):
        """
        Args:
            filename: path of the HDF5 file to write
            interval: how often to record a frame, see :class:`Interval`.
                Defaults to ``0``, the endpoints only — a trajectory is much
                larger than a log.
            store_forces: also store the forces of every frame of a
                relaxation. Doubles the file size.
            writer_kwargs: passed on to
                :class:`~schnetpack.dynamics.trajectory.TrajectoryWriter`,
                e.g. ``precision`` or ``buffer_size``
        """
        self.filename = filename
        self.interval = Interval(interval)
        self.store_forces = store_forces
        self.writer_kwargs = writer_kwargs
        self.writer: TrajectoryWriter | None = None

    def on_start(self, dynamics, batch) -> None:
        self.on_end()
        attrs = {"driver": type(dynamics).__name__}
        integrator = getattr(dynamics, "integrator", None)
        if integrator is not None:
            attrs["integrator"] = type(integrator).__name__
        fmax = getattr(dynamics, "fmax", None)
        if fmax is not None:
            attrs["fmax"] = fmax
        self.writer = TrajectoryWriter(
            self.filename,
            atomic_numbers=batch[properties.Z],
            n_atoms=batch[properties.n_atoms],
            pbc=batch.get(properties.pbc),
            store_forces=self.store_forces,
            attrs=attrs,
            **self.writer_kwargs,
        )

    def wants(self, step: int, final: bool) -> bool:
        return self.interval.due(step, final)

    def on_frame(self, frame: Frame) -> None:
        if self.writer is None:
            raise RuntimeError(
                "TrajectoryRecorder.on_frame was called before on_start; the "
                "trajectory file has not been created yet"
            )
        fields = {name: getattr(frame, attr) for name, attr in frame.trajectory_fields}
        self.writer.write(
            step=frame.step,
            positions=frame.positions,
            cell=frame.batch.get(properties.cell),
            **fields,
        )

    def on_end(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

"""Knowledge distillation: a student trained on its teacher's energy, forces
and curvature.

The teacher runs online, on the batch the student sees (ADR-0010). For the
curvature term both models are probed along the same random vector v: the
teacher supplies the curvature target ``H_t v``, the student its own
``H_s v``, both exact second derivatives. With v ~ N(0, 1) per component,
``E|(H_s - H_t) v|^2`` is the squared Frobenius distance of the two Hessians.

Everything here is pure PyTorch. :class:`~schnetpack.lightning.AtomisticTask`
distills through :func:`distillation_predictions`, and so does a hand-written
loop, which checks the setup once and assembles the loss with
:func:`~schnetpack.objectives.calculate_loss` (ADR-0025)::

    check_distillation_setup(outputs, model, teacher)
    for batch in loader:
        pred, targets = distillation_predictions(outputs, model, teacher, batch)
        loss = calculate_loss(outputs, pred, targets)
        ...

Evaluation needs gradients enabled too, since curvature is a derivative.
"""

import contextlib
import hashlib
import json
import logging
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.data.loader import AtomsLoader
from schnetpack.model.base import AtomisticModel
from schnetpack.model.utils import train_mode, train_mode_sensitive_modules
from schnetpack.objectives import (
    ModelOutput,
    UnsupervisedModelOutput,
    extract_targets,
    predict_without_postprocessing,
)
from schnetpack.train.teacher import TEACHER_PREFIX, TeacherWrapper
from schnetpack.transform import AddOffsets

__all__ = [
    "check_distillation_setup",
    "distillation_predictions",
    "student_stats_source",
    "TeacherStats",
]

log = logging.getLogger(__name__)

#: training structures the teacher runs on to fit the student's offsets, by
TEACHER_STATS_SIZE = 10_000


def draw_probe(
    positions: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A probe shaped like the positions: N(0, 1) per atom and Cartesian
    component, not normalized (ADR-0010 §2).

    With a generator the probe is drawn on cpu and moved over, so a seeded
    generator gives the same probes on every device.
    """
    if generator is None:
        return torch.randn_like(positions)
    probe = torch.randn(positions.shape, generator=generator, dtype=positions.dtype)
    return probe.to(positions.device)


def student_hvp(
    forces: torch.Tensor,
    positions: torch.Tensor,
    probe: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """The student's Hessian-vector product ``H_s v`` along ``probe``.

    Args:
        forces: the student's forces, computed from ``positions`` with a graph.
        positions: the positions the forces were differentiated by.
        probe: the probe, shaped like the positions.
        create_graph: keep the graph, so that a loss on the product can be
            backpropagated to the student's parameters. Needed in training only.
    """
    if forces.grad_fn is None:
        raise RuntimeError(
            "the student's forces carry no graph, so its curvature cannot be "
            "taken. SchNetPack's Forces build one only in train mode: run the "
            "student in train mode (see schnetpack.model.train_mode)."
        )
    # F = -dE/dR, so the vector-Jacobian product of F with v is -H v
    (minus_hvp,) = grad(
        forces,
        positions,
        grad_outputs=probe,
        create_graph=create_graph,
        retain_graph=True,
    )
    return -minus_hvp


def student_energy_offsets(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    energy_key: str = properties.energy,
) -> torch.Tensor:
    """The offsets the student's ``AddOffsets`` add to its energy, per structure.

    The student is compared without postprocessing, so an absolute teacher
    energy has to lose exactly these offsets first. Computed in float64: the
    offsets are about as large as the absolute energies.
    """
    n_atoms = batch[properties.n_atoms]
    inputs = {
        energy_key: torch.zeros(
            n_atoms.shape[0], dtype=torch.float64, device=n_atoms.device
        ),
        properties.n_atoms: n_atoms,
        properties.Z: batch[properties.Z],
        properties.idx_m: batch[properties.idx_m],
    }
    for module in getattr(model, "postprocessors", ()):
        if isinstance(module, AddOffsets) and module._property == energy_key:
            inputs = module(inputs)
    return inputs[energy_key]


def needs_teacher(outputs: Sequence[ModelOutput]) -> bool:
    """Whether any output is trained on a teacher target."""
    return any(_is_teacher_target(output) for output in outputs)


def _is_teacher_target(output: ModelOutput) -> bool:
    return output.target_property.startswith(TEACHER_PREFIX)


def needs_curvature(outputs: Sequence[ModelOutput]) -> bool:
    """Whether any output reads the student's Hessian-vector product."""
    return any(output.name == properties.hvp for output in outputs)


def check_distillation_setup(
    outputs: Sequence[ModelOutput],
    model: nn.Module,
    teacher: TeacherWrapper | None,
) -> None:
    """Refuse a setup that cannot distill, before any step runs."""
    if (needs_teacher(outputs) or needs_curvature(outputs)) and teacher is None:
        raise ValueError(
            f"the outputs refer to teacher targets or to {properties.hvp!r}, but "
            "no teacher was given"
        )
    if teacher is None:
        return
    unknown = sorted(
        {
            output.target_property
            for output in outputs
            if _is_teacher_target(output)
            and output.target_property not in teacher.target_keys
        }
    )
    if unknown:
        raise ValueError(
            f"the outputs refer to teacher targets {unknown}, but the teacher "
            f"supplies {list(teacher.target_keys)}: one target per student key "
            "it was given"
        )
    student_keys = (teacher.student_energy_key, teacher.student_force_key)
    model_outputs = getattr(model, "model_outputs", None)
    if model_outputs is not None:
        missing = [key for key in student_keys if key not in model_outputs]
        if missing:
            raise ValueError(
                f"the teacher was given student keys {missing}, but the student "
                f"outputs {sorted(model_outputs)}"
            )
    if needs_curvature(outputs):
        sensitive = train_mode_sensitive_modules(model)
        if sensitive:
            raise ValueError(
                f"the student contains train/eval-sensitive modules {sensitive}. "
                "It runs in train mode whenever its curvature is taken, "
                "validation included, so these would be active there."
            )


def distillation_predictions(
    outputs: Sequence[ModelOutput],
    model: AtomisticModel,
    teacher: TeacherWrapper,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator | None = None,
    create_graph: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Predictions and targets of one distillation step.

    Labels are read off the batch first and the teacher runs next, both before
    the student: a SchNetPack model writes its outputs into the batch it is
    given. The teacher's energy loses the student's offsets, since the student
    is compared without postprocessing. When an output reads ``hvp``, one probe
    is drawn for teacher and student alike, and the student runs in train mode
    so that its forces keep their graph.

    Args:
        outputs: the loss terms.
        model: the student.
        teacher: the teacher.
        batch: student batch.
        generator: generator to draw the probe from; None for the global one.
        create_graph: keep the graph of the student's curvature, as training
            needs.

    Returns:
        ``(pred, targets)``, ready for the loss.
    """
    targets = extract_targets(outputs, batch, skip=teacher.target_keys)
    curvature = needs_curvature(outputs)
    probe = draw_probe(batch[properties.R], generator) if curvature else None

    if needs_teacher(outputs):
        teacher_targets = teacher(batch, probe)
        energy_key = teacher.target_keys[0]
        energy = teacher_targets[energy_key] - student_energy_offsets(
            model, batch, teacher.student_energy_key
        )
        teacher_targets[energy_key] = energy.to(batch[properties.R].dtype)
        targets.update(teacher_targets)

    mode = train_mode(model) if curvature else contextlib.nullcontext()
    with mode:
        pred = predict_without_postprocessing(model, batch)
    if curvature:
        pred[properties.hvp] = student_hvp(
            pred[teacher.student_force_key], batch[properties.R], probe, create_graph
        )
    return pred, targets


def student_stats_source(
    outputs: Sequence[ModelOutput],
    datamodule,
    teacher: TeacherWrapper,
    device: torch.device | str | None = None,
    max_structures: int | None = TEACHER_STATS_SIZE,
):
    """The stats source the student's offsets are fitted to (ADR-0017).

    A label term on the student's energy key means the data pipeline removes
    offsets fitted to those labels from them. The student's offsets must be the
    same ones, so the datamodule answers, and the teacher's energy loses them
    too. Without such a term the teacher's energies stand in for labels: a
    :class:`TeacherStats` on at most ``max_structures`` training structures.
    """
    energy_key = teacher.student_energy_key
    if any(
        not isinstance(output, UnsupervisedModelOutput)
        and output.target_property == energy_key
        for output in outputs
    ):
        return datamodule
    return TeacherStats(
        datamodule, teacher, device=device, max_structures=max_structures
    )


class TeacherStats:
    """A stats source that reads the student's energy statistics off the teacher.

    ``AddOffsets`` initializes from a stats source (``get_stats`` and
    ``get_atomrefs``), normally the datamodule, which computes them from energy
    labels. A distillation dataset may have none, so this source runs the
    teacher instead (ADR-0010 §6, ADR-0017): once, on the first query for the
    student's energy, ``teacher.student_energy_key``, and on a seeded random
    subset of at most ``max_structures`` training structures, the same on
    every rank. Queries for any other property go to the datamodule.

    The results are stored through the datamodule's stats provider, keyed by
    :attr:`fingerprint`, so that reruns, resumes and other ranks read them
    instead of running the teacher. Without a provider or a stats file,
    nothing is stored.

    ``remove_atomref`` removes the atomrefs this source handed out for the
    property, or else ones estimated from the teacher.

    Args:
        datamodule: provides ``train_dataset``, ``batch_size`` and
            ``num_workers``, answers queries for other properties, and stores
            results through its ``provider``, if it has one.
        teacher: the teacher.
        device: device to run the teacher on for this pass; None for the
            batches' own.
        max_structures: training structures to run the teacher on; None for
            the whole split.
        seed: seed of the random subset.
        z_max: size of the atomref tensor, as in ``AddOffsets``.
    """

    def __init__(
        self,
        datamodule,
        teacher: TeacherWrapper,
        device: torch.device | str | None = None,
        max_structures: int | None = TEACHER_STATS_SIZE,
        seed: int = 0,
        z_max: int = 100,
    ):
        self.datamodule = datamodule
        self.teacher = teacher
        self.energy_key = teacher.student_energy_key
        self.device = device
        self.max_structures = max_structures
        self.seed = seed
        self.z_max = z_max
        self._energies: torch.Tensor | None = None
        self._n_atoms: torch.Tensor | None = None
        #: atoms of each element per structure; float32 is exact for counts
        self._counts: torch.Tensor | None = None
        #: atomrefs handed out per property, with where they came from
        self._atomrefs: dict[str, tuple[torch.Tensor, str]] = {}
        self._fingerprint: str | None = None

    @property
    def fingerprint(self) -> str:
        """What the stored results are valid for: the contents of the
        teacher's model file, its configuration and family, and the subset."""
        if self._fingerprint is None:
            config = self.teacher.__getstate__()
            model_path = config.pop("model_path", None)
            payload = {
                "family": type(self.teacher).__qualname__,
                "config": config,
                "max_structures": self.max_structures,
                "seed": self.seed,
            }
            if model_path is not None:
                with open(model_path, "rb") as file:
                    payload["model"] = hashlib.file_digest(file, "sha256").hexdigest()
            text = json.dumps(payload, sort_keys=True, default=str)
            self._fingerprint = hashlib.sha256(text.encode()).hexdigest()[:16]
        return self._fingerprint

    def _stored(self, entry: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        provider = getattr(self.datamodule, "provider", None)
        if provider is None:
            return compute()
        return provider.read_or_compute(f"teacher:{self.fingerprint}:{entry}", compute)

    def _teacher_pass(self) -> None:
        if self._energies is not None:
            return
        dataset = self.datamodule.train_dataset
        if self.max_structures is not None and self.max_structures < len(dataset):
            generator = torch.Generator().manual_seed(self.seed)
            subset = torch.randperm(len(dataset), generator=generator)
            subset = sorted(subset[: self.max_structures].tolist())
            dataset = dataset.subset(subset, split=dataset.split)
        log.info(
            f"Running the teacher on {len(dataset)} training structures for the "
            "student's energy offsets"
        )
        loader = AtomsLoader(
            dataset,
            batch_size=self.datamodule.batch_size,
            shuffle=False,
            num_workers=self.datamodule.num_workers,
        )
        energy_key = self.teacher.target_keys[0]
        energies, n_atoms, counts = [], [], []
        for batch in loader:
            if self.device is not None:
                batch = {key: value.to(self.device) for key, value in batch.items()}
            energies.append(self.teacher(batch)[energy_key].cpu())
            numbers = batch[properties.Z].cpu()
            idx_m = batch[properties.idx_m].cpu()
            count = torch.zeros(len(batch[properties.n_atoms]), self.z_max)
            count.index_put_(
                (idx_m, numbers),
                torch.ones_like(numbers, dtype=count.dtype),
                accumulate=True,
            )
            counts.append(count)
            n_atoms.append(batch[properties.n_atoms].cpu())
        self._energies = torch.cat(energies)
        self._n_atoms = torch.cat(n_atoms).to(torch.float64)
        self._counts = torch.cat(counts)

    def _estimate_atomrefs(self) -> np.ndarray:
        self._teacher_pass()
        present = self._counts.sum(0) > 0
        design = self._counts[:, present].to(torch.float64)
        # lstsq copes with compositions that always occur in fixed ratios
        solution = torch.linalg.lstsq(design, self._energies[:, None]).solution
        atomref = torch.zeros(self.z_max, dtype=torch.float64)
        atomref[present] = solution[:, 0]
        return atomref.numpy()

    def get_atomrefs(
        self, property: str, is_extensive: bool, estimate: bool = True
    ) -> dict[str, torch.Tensor]:
        if property != self.energy_key:
            return self.datamodule.get_atomrefs(property, is_extensive, estimate)
        if not estimate:
            atomrefs = self.datamodule.get_atomrefs(
                property, is_extensive, estimate=False
            )
            self._atomrefs[property] = (
                atomrefs[property].to(torch.float64),
                "dataset",
            )
            return atomrefs
        if not is_extensive:
            raise ValueError(
                "the teacher's energy is extensive; atomrefs for an intensive "
                "energy cannot be estimated from it"
            )

        stored = self._stored(
            "atomrefs:" + json.dumps([property]), self._estimate_atomrefs
        )
        atomref = torch.from_numpy(stored).to(torch.float64)
        self._atomrefs[property] = (atomref, "estimated")
        return {property: atomref}

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if property != self.energy_key:
            return self.datamodule.get_stats(property, divide_by_atoms, remove_atomref)

        if remove_atomref and property not in self._atomrefs:
            self.get_atomrefs(property, is_extensive=True)
        source = self._atomrefs[property][1] if remove_atomref else None

        def compute() -> np.ndarray:
            self._teacher_pass()
            values = self._energies
            if remove_atomref:
                atomref = self._atomrefs[property][0]
                values = values - self._counts.to(torch.float64) @ atomref
            if divide_by_atoms:
                values = values / self._n_atoms
            return np.array([values.mean().item(), values.std(correction=0).item()])

        entry = json.dumps([property, divide_by_atoms, remove_atomref, source])
        mean, std = self._stored("stats:" + entry, compute)
        return (
            torch.tensor(mean, dtype=torch.float64),
            torch.tensor(std, dtype=torch.float64),
        )

"""Knowledge distillation: a student trained on its teacher's energy, forces
and curvature.

The teacher runs online, on the batch the student sees. For the
curvature term both models are probed along the same random vector v: the
teacher supplies the curvature target ``H_t v``, the student its own
``H_s v``, both exact second derivatives. With v ~ N(0, 1) per component,
``E|(H_s - H_t) v|^2`` is the squared Frobenius distance of the two Hessians.

Both products are taken through the forces, as ``-(dF/dR)^T v`` (see
:func:`~schnetpack.train.teacher.curvature_product`), so a model that predicts its
forces directly, not as -dE/dR, works as student and as teacher alike: its
force Jacobian stands in for the negative Hessian, and the term is the squared
Frobenius distance of the two force Jacobians.

The loss terms say which student output each teacher target trains::

    ModelOutput(name="energy", target_property=properties.teacher_energy, ...)

The teacher's energy is relative to its own offsets per element, and a student
trained on it adds the same offsets: :meth:`Distillation.setup` writes them
into the student's ``AddOffsets``. Both are then compared in one frame.

Everything here is pure PyTorch. A :class:`Distillation` checks the setup once,
when it is built; :class:`~schnetpack.lightning.AtomisticTask` distills through
one, and so does a hand-written loop, which assembles the loss with
:func:`~schnetpack.objectives.calculate_loss`::

    distillation = Distillation(outputs, model, teacher)
    distillation.setup(datamodule)
    for batch in loader:
        pred, targets = distillation.step(batch)
        loss = calculate_loss(outputs, pred, targets)
        ...

Evaluation needs gradients enabled too (see :attr:`Distillation.needs_grad`),
since curvature is a derivative.
"""

import contextlib
from collections.abc import Sequence

import torch
from torch import nn

from schnetpack import properties
from schnetpack.model.base import AtomisticModel
from schnetpack.model.utils import (
    forces_need_train_mode,
    train_mode,
    train_mode_sensitive_modules,
)
from schnetpack.objectives import (
    ModelOutput,
    UnsupervisedModelOutput,
    extract_targets,
    predict_without_postprocessing,
)
from schnetpack.train.teacher import TARGET_KEYS, TeacherWrapper, curvature_product
from schnetpack.transform import AddOffsets

__all__ = ["Distillation"]


def draw_probe(
    positions: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A probe shaped like the positions: N(0, 1) per atom and Cartesian
    component, not normalized: a per-component loss then weighs the
    curvature of every atom alike, as it does forces, whatever the
    structure's size.

    With a generator the probe is drawn on cpu and moved over, so a seeded
    generator gives the same probes on every device.
    """
    if generator is None:
        return torch.randn_like(positions)
    probe = torch.randn(positions.shape, generator=generator, dtype=positions.dtype)
    return probe.to(positions.device)


def needs_teacher(outputs: Sequence[ModelOutput]) -> bool:
    """Whether any output is trained on a teacher target."""
    return any(output.target_property in TARGET_KEYS for output in outputs)


def needs_curvature(outputs: Sequence[ModelOutput]) -> bool:
    """Whether any output reads the student's curvature."""
    return any(output.name == properties.curvature for output in outputs)


def student_key(outputs: Sequence[ModelOutput], target: str) -> str | None:
    """The student output trained on the teacher target ``target``; None if no
    output is.

    Raises:
        ValueError: if outputs of different names are trained on it.
    """
    names = sorted(
        {output.name for output in outputs if output.target_property == target}
    )
    if len(names) > 1:
        raise ValueError(
            f"outputs {names} are all trained on {target!r}; one student output "
            "per teacher target"
        )
    return names[0] if names else None


class Distillation:
    """A student distilled from a teacher, on a set of loss terms.

    Built once, before any step: it refuses a setup that cannot distill.
    :meth:`setup` then initializes the student, and each step asks
    :meth:`step` for its predictions and targets.

    The student's keys are read off the loss terms: its energy is the output
    trained on ``teacher_energy``, its forces the output trained on
    ``teacher_forces``, and its curvature is taken through those forces.

    A plain object rather than an ``nn.Module``: it holds the outputs, student
    and teacher by reference, and registers none of them.

    Args:
        outputs: the loss terms.
        model: the student.
        teacher: the teacher.
        probe_seed: seed of the probes outside training; :meth:`reset_probes`
            goes back to it, so that every evaluation sees the same probes.
            Training draws from the global generator.

    Raises:
        ValueError: if several student outputs are trained on one teacher
            target; if a curvature term has no term on ``teacher_forces`` to
            name the student's forces; if the student lacks an output a term
            names; if the student's energy is trained on labels and on the
            teacher at once, or lacks an ``AddOffsets`` with atomrefs to hold
            the teacher's offsets; or if the student must run in train mode
            for its curvature but contains train/eval-sensitive modules.
    """

    def __init__(
        self,
        outputs: Sequence[ModelOutput],
        model: AtomisticModel,
        teacher: TeacherWrapper,
        probe_seed: int = 0,
    ):
        self.outputs = outputs
        self.model = model
        self.teacher = teacher
        self.probe_seed = probe_seed
        #: the student's outputs trained on the teacher's energy and forces
        self.energy_key = student_key(outputs, properties.teacher_energy)
        self.force_key = student_key(outputs, properties.teacher_forces)
        self._generator = torch.Generator().manual_seed(probe_seed)
        self._check()

    @classmethod
    def optional(
        cls,
        outputs: Sequence[ModelOutput],
        model: AtomisticModel,
        teacher: TeacherWrapper | None,
        probe_seed: int = 0,
    ) -> "Distillation | None":
        """A distillation if there is a teacher, else None.

        Raises:
            ValueError: if there is no teacher, but the outputs need one.
        """
        if teacher is not None:
            return cls(outputs, model, teacher, probe_seed)
        if needs_teacher(outputs) or needs_curvature(outputs):
            raise ValueError(
                f"the outputs refer to teacher targets or to "
                f"{properties.curvature!r}, but no teacher was given"
            )
        return None

    @property
    def needs_grad(self) -> bool:
        """Whether evaluation needs gradients enabled: for the student's own
        derivatives, such as forces taken as -dE/dR, or for its curvature."""
        derivatives = getattr(self.model, "required_derivatives", None)
        return bool(derivatives) or needs_curvature(self.outputs)

    def _check(self) -> None:
        curvature = needs_curvature(self.outputs)
        if curvature and self.force_key is None:
            raise ValueError(
                f"the student's curvature is taken through its forces, but no "
                f"output is trained on {properties.teacher_forces!r} to name "
                "them. Add one; a loss_weight of 0 keeps it out of the loss."
            )
        model_outputs = getattr(self.model, "model_outputs", None)
        if model_outputs is not None:
            missing = [
                key
                for key in (self.energy_key, self.force_key)
                if key is not None and key not in model_outputs
            ]
            if missing:
                raise ValueError(
                    f"the outputs train student outputs {missing} on the teacher, "
                    f"but the student outputs {sorted(model_outputs)}"
                )
        if self.energy_key is not None:
            labelled = [
                output.target_property
                for output in self.outputs
                if output.name == self.energy_key
                and not isinstance(output, UnsupervisedModelOutput)
                and output.target_property not in TARGET_KEYS
            ]
            if labelled:
                raise ValueError(
                    f"{self.energy_key!r} is trained on the teacher and on labels "
                    f"{labelled} at once. The teacher's energy is relative to its "
                    "own offsets, the labels to those fitted by the data "
                    "pipeline; train it on one of them."
                )
            self._energy_offsets()
        if curvature and forces_need_train_mode(self.model):
            sensitive = train_mode_sensitive_modules(self.model)
            if sensitive:
                raise ValueError(
                    f"the student contains train/eval-sensitive modules "
                    f"{sensitive}. It runs in train mode whenever its curvature "
                    "is taken, validation included, so these would be active "
                    "there."
                )

    def _energy_offsets(self) -> AddOffsets:
        """The student's ``AddOffsets`` on its energy, which takes the
        teacher's offsets."""
        modules = [
            module
            for module in getattr(self.model, "postprocessors", ())
            if isinstance(module, AddOffsets) and module._property == self.energy_key
        ]
        if len(modules) != 1 or not modules[0].add_atomrefs:
            raise ValueError(
                f"the student's {self.energy_key!r} is trained on the teacher's "
                "energy, which is relative to the teacher's offsets per element. "
                "The student adds the same offsets, so it needs exactly one "
                f"AddOffsets on {self.energy_key!r}, with add_atomrefs=True."
            )
        return modules[0]

    def setup(self, datamodule) -> None:
        """Initialize the student: the teacher's offsets into the
        ``AddOffsets`` on its energy, if a term trains it on the teacher's,
        and the rest from ``datamodule`` (see
        :meth:`~schnetpack.model.AtomisticModel.initialize_transforms`)."""
        if self.energy_key is not None:
            module = self._energy_offsets()
            size = len(module.atomref)
            offsets = self.teacher.energy_offsets()
            if offsets[size:].any():
                raise ValueError(
                    f"the teacher has offsets for atomic numbers from {size} on, "
                    f"beyond the student's AddOffsets (zmax={size})"
                )
            offsets = offsets[:size]
            offsets = nn.functional.pad(offsets, (0, size - len(offsets)))
            module.set_offsets(offsets, torch.zeros(1, dtype=offsets.dtype))
        self.model.initialize_transforms(datamodule)

    def reset_probes(self) -> None:
        """Reseed the probes drawn outside training."""
        self._generator.manual_seed(self.probe_seed)

    def step(
        self, batch: dict[str, torch.Tensor], train: bool = True
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Predictions and targets of one distillation step.

        Labels are read off the batch first and the teacher runs next, both
        before the student: a SchNetPack model writes its outputs into the
        batch it is given. When an output reads ``curvature``, one probe is
        drawn for teacher and student alike, and the student's positions are
        made to require grad (False by default for directly predicted forces).
        A student whose forces are a position derivative runs in train mode
        then, so that they keep their graph.

        Args:
            batch: student batch.
            train: a training step: the probe comes from the global generator
                and the graph of the student's curvature is kept. Otherwise
                the probe comes from the seeded generator (see
                :meth:`reset_probes`), and no graph is kept.

        Returns:
            ``(pred, targets)``, ready for the loss.
        """
        model = self.model
        targets = extract_targets(self.outputs, batch, skip=TARGET_KEYS)
        curvature = needs_curvature(self.outputs)
        probe = None
        if curvature:
            generator = None if train else self._generator
            probe = draw_probe(batch[properties.R], generator)

        if needs_teacher(self.outputs):
            targets.update(self.teacher(batch, probe))

        switch = curvature and forces_need_train_mode(model)
        mode = train_mode(model) if switch else contextlib.nullcontext()
        with mode:
            if curvature:
                batch[properties.R].requires_grad_()
            pred = predict_without_postprocessing(model, batch)
        if curvature:
            pred[properties.curvature] = curvature_product(
                pred[self.force_key],
                batch[properties.R],
                probe,
                create_graph=train,
                model="student",
            )
        return pred, targets

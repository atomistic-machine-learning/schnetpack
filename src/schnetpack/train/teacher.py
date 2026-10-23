"""Teachers for knowledge distillation.

A teacher is the frozen model a student is distilled from. It may work in other
units than the student, with another cutoff and -- if it is not a SchNetPack
model -- with another input and output format. :class:`TeacherWrapper` is the
boundary between a teacher and training: it takes a student batch in the
student's units and returns the teacher targets in the student's units, under
the fixed keys ``properties.teacher_energy``, ``teacher_forces`` and
``teacher_curvature``. It knows nothing of the student: which student output
each target trains is stated by the loss terms.

Each kind of teacher model is a subclass that says how the teacher is loaded
and how its output for a batch is computed.
"""

import contextlib
import os
import warnings
from typing import Any

import torch
from torch import nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.data import prune_neighbors
from schnetpack.model.mace import MaceModel, energy_shift, load_mace
from schnetpack.model.utils import (
    forces_need_train_mode,
    train_mode,
    train_mode_sensitive_modules,
)
from schnetpack.transform import AddOffsets
from schnetpack.units import convert_units
from schnetpack.utils import as_dtype
from schnetpack.utils.compatibility import load_model

__all__ = [
    "TeacherWrapper",
    "SchNetPackTeacher",
    "MaceTeacher",
]

#: the targets a teacher supplies: energy, forces, curvature
TARGET_KEYS = (
    properties.teacher_energy,
    properties.teacher_forces,
    properties.teacher_curvature,
)


def curvature_product(
    forces: torch.Tensor,
    positions: torch.Tensor,
    probe: torch.Tensor,
    create_graph: bool = False,
    model: str = "model",
) -> torch.Tensor:
    """The curvature of a model along ``probe``: ``-(dF/dR)^T v``, a negated
    vector-Jacobian product, taken in one reverse-mode pass.

    For forces ``F = -dE/dR`` the Jacobian is the negative Hessian, which is
    symmetric, so this is the Hessian-vector product ``H v``. For forces
    predicted directly the Jacobian is not symmetric, and this is neither a
    Hessian product nor the Jacobian-vector product ``-(dF/dR) v``; the loss
    does not mind, since ``E|(J_s - J_t)^T v|^2 = |J_s - J_t|_F^2`` as well.
    Teacher and student both take their curvature here, so that the loss
    compares like with like.

    Args:
        forces: forces computed from ``positions``, with a graph.
        positions: the positions the forces were computed from.
        probe: the probe, shaped like the positions.
        create_graph: keep the graph of the product, so that a loss on it can
            be backpropagated.
        model: which model the forces are from, for the error message.
    """
    if forces.grad_fn is None:
        raise RuntimeError(
            f"the {model}'s forces are not differentiable with respect to the "
            "positions, so its curvature cannot be taken. This happens when they "
            "were computed without a graph (SchNetPack's Forces build one only "
            "in train mode), when the model was exported for inference, or when "
            "they do not depend on the positions."
        )
    (vjp,) = grad(
        forces,
        positions,
        grad_outputs=probe,
        create_graph=create_graph,
        retain_graph=True,
    )
    return -vjp


#: accepted values of ``SchNetPackTeacher(model_format=...)``; ``"auto"`` tries
#: TorchScript first and falls back to a pickled model
MODEL_FORMATS = ("auto", "torch", "torchscript")

#: largest atomic number the offset table covers
MAX_Z = 118


class TeacherWrapper:
    """A frozen teacher, presented to training in the student's units.

    ``teacher(batch, probe)`` returns ``teacher_energy``, ``teacher_forces`` and,
    given a probe, ``teacher_curvature`` (see :data:`TARGET_KEYS`), all in the
    batch's dtype, detached; the batch is never written to.

    The energy is relative to the teacher's own offsets per element
    (:meth:`teacher_offsets`): ``E - sum_i offset[Z_i]``, taken in float64 before
    the cast, so that absolute energies do not cost it digits. A student trained
    on it adds the same offsets, :meth:`energy_offsets`, in its ``AddOffsets``
    (see :class:`~schnetpack.train.Distillation`), so both are compared in one
    frame.

    The wrapper is a plain object rather than an ``nn.Module``, so a task
    holding it does not register it: the teacher stays out of ``state_dict()``,
    ``parameters()`` and the optimizer. It pickles as its configuration -- its
    public attributes; private ones are runtime state that :meth:`load`
    rebuilds -- and loads the teacher again when unpickled, so checkpoints stay
    small. Being frozen, it is shared rather than copied by ``copy.deepcopy``.

    Subclasses implement :meth:`load` and :meth:`teacher_output`, and
    :meth:`teacher_offsets` if the model's energy offsets are known.

    Args:
        teacher_energy_unit: energy unit the teacher works in.
        teacher_distance_unit: length unit the teacher works in.
        student_energy_unit: energy unit of the student, and of the targets.
        student_distance_unit: length unit of the student's batches.
        cutoff: the teacher's cutoff, in the teacher's length unit. Teacher
            and student share the batch's one neighbor list, built by the data
            pipeline to cover both cutoffs (see
            :class:`~schnetpack.transform.DistillationNeighborList`); the
            teacher prunes it to ``cutoff`` with
            :func:`~schnetpack.data.prune_neighbors`, triples included.
            None passes the list through as it is -- right for a list
            built at the teacher's own cutoff.
        dtype: floating dtype the teacher runs in.
    """

    def __init__(
        self,
        teacher_energy_unit: str | float = "eV",
        teacher_distance_unit: str | float = "Ang",
        student_energy_unit: str | float = "eV",
        student_distance_unit: str | float = "Ang",
        cutoff: float | None = None,
        dtype: torch.dtype | str = torch.float32,
    ):
        self.teacher_energy_unit = teacher_energy_unit
        self.teacher_distance_unit = teacher_distance_unit
        self.student_energy_unit = student_energy_unit
        self.student_distance_unit = student_distance_unit
        self.cutoff = cutoff
        self.dtype = as_dtype(dtype)
        #: teacher -> student
        self.position_conversion = convert_units(
            teacher_distance_unit, student_distance_unit
        )
        self.energy_conversion = convert_units(teacher_energy_unit, student_energy_unit)
        self.force_conversion = self.energy_conversion / self.position_conversion
        self.hessian_conversion = self.force_conversion / self.position_conversion
        self._setup()

    # --------------------------------------------------------- subclass hooks

    def load(self) -> Any:
        """Load the teacher. Called at construction and after unpickling."""
        raise NotImplementedError

    def teacher_output(
        self, inputs: dict[str, torch.Tensor], create_graph: bool
    ) -> dict[str, torch.Tensor]:
        """The teacher's ``{"energy": (n_structures,), "forces": (n_atoms, 3)}``
        in teacher units, the energy absolute.

        ``inputs`` is a SchNetPack batch in the teacher's units and dtype, with
        the teacher's neighbor list. Its positions are the tensor the forces are
        differentiated by: a translation must build on them, not copy them.
        ``create_graph`` is True when a curvature target is asked for; the
        forces must then come back differentiable with respect to those
        positions, however the subclass achieves that.
        """
        raise NotImplementedError

    def teacher_offsets(self) -> torch.Tensor:
        """The energy the teacher adds per atom of each element, indexed by
        atomic number, in float64 and the teacher's energy unit; zeros if the
        subclass cannot tell. A mean per atom is folded in: it is added to every
        element. Called once, when first needed."""
        return torch.zeros(1, dtype=torch.float64)

    # ----------------------------------------------------------------- targets

    def energy_offsets(self) -> torch.Tensor:
        """The teacher's offsets per element, in float64 and the student's
        energy unit: the atomrefs a student trained on ``teacher_energy`` adds."""
        return self._offsets_table() * self.energy_conversion

    def __call__(
        self, batch: dict[str, torch.Tensor], probe: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Teacher targets for the structures in ``batch``.

        Args:
            batch: student batch, in the student's units.
            probe: ``(n_atoms, 3)`` vector along which to take the teacher's
                curvature; None for energy and forces only.
        """
        student_dtype = batch[properties.R].dtype
        inputs = self._teacher_inputs(batch)
        positions = inputs[properties.R]

        create_graph = probe is not None
        curvature = None
        with torch.enable_grad():
            output = self.teacher_output(inputs, create_graph)
            energy = output[properties.energy]
            forces = output[properties.forces]
            if probe is not None:
                curvature = curvature_product(
                    forces, positions, probe.to(forces.dtype), model="teacher"
                )

        energy = energy.detach().to(torch.float64) - self._offsets_of(batch)
        targets = {
            properties.teacher_energy: (energy * self.energy_conversion).to(
                student_dtype
            ),
            properties.teacher_forces: (forces.detach() * self.force_conversion).to(
                student_dtype
            ),
        }
        if curvature is not None:
            targets[properties.teacher_curvature] = (
                curvature * self.hessian_conversion
            ).to(student_dtype)
        return targets

    def _offsets_of(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """The teacher's offsets summed per structure, in its energy unit."""
        table = self._offsets_table().to(batch[properties.Z].device)
        idx_m = batch[properties.idx_m]
        per_structure = torch.zeros(
            len(batch[properties.n_atoms]), dtype=torch.float64, device=idx_m.device
        )
        return per_structure.index_add(0, idx_m, table[batch[properties.Z]])

    def _offsets_table(self) -> torch.Tensor:
        if self._offsets is None:
            offsets = self.teacher_offsets()
            # index safety: every element has an entry, zero if the subclass has none
            size = max(len(offsets), MAX_Z + 1)
            self._offsets = nn.functional.pad(offsets, (0, size - len(offsets)))
        return self._offsets

    def _teacher_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """A copy of ``batch`` in the teacher's units, dtype and device, with the
        neighbor list pruned to the teacher's cutoff, and with positions that are
        a fresh leaf requiring grad."""
        self._to_device(batch[properties.R].device)
        inputs = {
            key: value.to(self.dtype) if value.is_floating_point() else value
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        for key in (properties.R, properties.cell, properties.offsets):
            if key in inputs:
                inputs[key] = inputs[key].detach() / self.position_conversion
        inputs.pop(properties.Rij, None)

        if self.cutoff is not None:
            inputs.update(prune_neighbors(inputs, inputs[properties.R], self.cutoff))

        inputs[properties.R] = inputs[properties.R].detach().requires_grad_()
        return inputs

    # ------------------------------------------------------ state and devices

    def _setup(self) -> None:
        self.model = self.load()
        self._device: torch.device | None = None
        self._offsets: torch.Tensor | None = None

    def _to_device(self, device: torch.device) -> None:
        if device != self._device:
            if isinstance(self.model, nn.Module):
                self.model.to(device)
            self._device = device

    def __getstate__(self) -> dict[str, Any]:
        # the teacher is frozen and lives on disk: store how to load it instead;
        # private attributes are runtime state that load() rebuilds
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "model" and not key.startswith("_")
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._setup()

    def __deepcopy__(self, memo: dict) -> "TeacherWrapper":
        # frozen, so copies may share it; Lightning deep-copies hyperparameters,
        # which would otherwise load the teacher from disk a second time
        return self


class SchNetPackTeacher(TeacherWrapper):
    """A SchNetPack model as teacher, saved with ``torch.save`` or as TorchScript.

    The model runs as it predicts: its postprocessing stays on, so its energies
    are absolute. It is kept in eval mode, and switched to train mode while a
    curvature target is taken so that its ``Forces`` keep their graph. A model
    that predicts its forces directly needs no switch and stays in eval mode
    (see :func:`~schnetpack.model.utils.forces_need_train_mode`).

    Args:
        model_path: path to the saved model.
        model_format: ``"torch"`` for a pickled model (SchNetPack 2.x included),
            ``"torchscript"`` for an archive from ``torch.jit.save``, or
            ``"auto"`` to try TorchScript first.
        teacher_energy_key: the model's energy output.
        teacher_force_key: the model's force output.
        **kwargs: units, cutoff and dtype; see :class:`TeacherWrapper`.
    """

    def __init__(
        self,
        model_path: str,
        model_format: str = "auto",
        teacher_energy_key: str = properties.energy,
        teacher_force_key: str = properties.forces,
        **kwargs,
    ):
        if model_format not in MODEL_FORMATS:
            raise ValueError(
                f"unknown model_format {model_format!r}, expected one of "
                f"{MODEL_FORMATS}"
            )
        self.model_path = model_path
        self.model_format = model_format
        self.teacher_energy_key = teacher_energy_key
        self.teacher_force_key = teacher_force_key
        super().__init__(**kwargs)

    def load(self) -> nn.Module:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"no teacher model at {self.model_path!r}")

        if self.model_format == "torchscript":
            model = torch.jit.load(self.model_path, map_location="cpu")
        elif self.model_format == "torch":
            model = load_model(self.model_path, device="cpu")
        else:
            try:
                model = torch.jit.load(self.model_path, map_location="cpu")
            except RuntimeError:
                model = load_model(self.model_path, device="cpu")

        model = model.to(self.dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        try:
            model.do_postprocessing = True
        except (AttributeError, RuntimeError) as err:
            warnings.warn(
                f"could not turn on the teacher's postprocessing ({err}); its "
                "energies may lack its offsets.",
                stacklevel=2,
            )

        sensitive = train_mode_sensitive_modules(model)
        if sensitive and forces_need_train_mode(model):
            warnings.warn(
                f"the teacher contains train/eval-sensitive modules {sensitive}. "
                "It runs in train mode while curvature targets are taken, so "
                "these are active then.",
                stacklevel=2,
            )
        return model

    def teacher_output(
        self, inputs: dict[str, torch.Tensor], create_graph: bool
    ) -> dict[str, torch.Tensor]:
        # its Forces keep their graph only in train mode; direct forces need none
        mode = (
            train_mode(self.model)
            if create_graph and forces_need_train_mode(self.model)
            else contextlib.nullcontext()
        )
        with mode:
            output = self.model(inputs)
        return {
            properties.energy: output[self.teacher_energy_key],
            properties.forces: output[self.teacher_force_key],
        }

    def teacher_offsets(self) -> torch.Tensor:
        """The atomrefs plus the mean per atom of the model's ``AddOffsets`` on
        its energy, pickled or scripted. A model without one adds nothing, and
        gets a warning: the student then learns the absolute energies the
        teacher's network predicts."""
        offsets = torch.zeros(1, dtype=torch.float64)
        found = False
        postprocessors = getattr(self.model, "postprocessors", None)
        for module in postprocessors.children() if postprocessors is not None else ():
            is_offsets = isinstance(module, AddOffsets) or (
                getattr(module, "original_name", None) == "AddOffsets"
            )
            if not is_offsets or module._property != self.teacher_energy_key:
                continue
            if not module.is_extensive:
                raise ValueError(
                    f"the teacher's offsets on {self.teacher_energy_key!r} are "
                    "intensive; an energy's are extensive"
                )
            found = True
            atomref = module.atomref.detach().to(torch.float64).cpu()
            mean = module.mean.detach().to(torch.float64).cpu().reshape(-1)[0]
            per_element = torch.zeros_like(atomref)
            if module.add_atomrefs:
                per_element += atomref
            if module.add_mean:
                per_element += mean
            size = max(len(offsets), len(per_element))
            offsets = nn.functional.pad(offsets, (0, size - len(offsets)))
            offsets[: len(per_element)] += per_element
        if not found:
            warnings.warn(
                f"the teacher has no AddOffsets on {self.teacher_energy_key!r}, so "
                "the student starts from zero offsets and learns the absolute "
                "energies the teacher's network predicts",
                stacklevel=2,
            )
        return offsets


class MaceTeacher(SchNetPackTeacher):
    """A MACE model as teacher (e.g. MACE-OFF), run through
    :class:`~schnetpack.model.MaceModel`.

    The teacher reads TorchScript archives, which load without ``mace-torch``
    and run compiled. Building one needs ``schnetpack[mace]`` installed::

        from schnetpack.interfaces.mace_interface import compile_mace_off

        path = compile_mace_off("medium")

    The teacher works in eV and Å. Its energies are absolute: its per-element
    reference energies (E0s) are added in float64, read from the model before
    it is cast to ``dtype``, so a float32 teacher keeps the digits of its
    energies. Its forces keep their graph in train mode, which a curvature
    target switches on. MACE-OFF is licensed for academic use only (ASL).

    Args:
        model_path: path to the TorchScript archive.
        **kwargs: units, cutoff and dtype; see :class:`TeacherWrapper`. The
            cutoff must not be smaller than the model's ``r_max`` (5 Å for
            MACE-OFF23), or pairs the model sees would be pruned away.
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)

    def load(self) -> MaceModel:
        model = MaceModel(load_mace(self.model_path), dtype=self.dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if self.cutoff is not None and self.cutoff < model.r_max:
            raise ValueError(
                f"the teacher's cutoff {self.cutoff} is below the model's r_max "
                f"{model.r_max}; pruning would drop pairs the model sees"
            )
        return model

    def teacher_offsets(self) -> torch.Tensor:
        """The E0s plus the shift of the model's ``ScaleShiftBlock``: the
        energy MACE adds per atom of each element."""
        return super().teacher_offsets() + energy_shift(self.model.core)

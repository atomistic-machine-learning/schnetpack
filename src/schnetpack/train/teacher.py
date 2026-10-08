"""Teachers for knowledge distillation.

A teacher is the frozen model a student is distilled from. It may work in other
units than the student, with another cutoff and -- if it is not a SchNetPack
model -- with another input and output format. :class:`TeacherWrapper` is the
boundary between a teacher and training: it takes a student batch in the
student's units and returns the teacher targets in the student's units,
named after the student's keys.

Each teacher family is a subclass that says how the teacher is loaded and how
its output for a batch is computed.
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
from schnetpack.model.utils import train_mode, train_mode_sensitive_modules
from schnetpack.units import convert_units
from schnetpack.utils import as_dtype
from schnetpack.utils.compatibility import load_model

__all__ = [
    "teacher_key",
    "TeacherWrapper",
    "SchNetPackTeacher",
    "MaceTeacher",
]

#: reserved for teacher targets; no label or model output starts with it
TEACHER_PREFIX = "teacher_"


def teacher_key(student_key: str) -> str:
    """The key of the teacher target for the student's ``student_key``."""
    return TEACHER_PREFIX + student_key


#: accepted values of ``SchNetPackTeacher(model_format=...)``; ``"auto"`` tries
#: TorchScript first and falls back to a pickled model
MODEL_FORMATS = ("auto", "torch", "torchscript")


class TeacherWrapper:
    """A frozen teacher, presented to training in the student's units.

    ``teacher(batch, probe)`` returns the teacher's energy per structure in
    float64 -- teacher energies are absolute and would lose digits in float32 --
    and its forces and, given a probe, ``teacher_hvp`` in the batch's dtype.
    Each target is named after the student's key it is compared with:
    ``teacher_<student_energy_key>``, ``teacher_<student_force_key>`` (see
    :func:`teacher_key` and :attr:`target_keys`). All of them are detached, and
    the batch is never written to.

    The wrapper is a plain object rather than an ``nn.Module``, so a task
    holding it does not register it: the teacher stays out of ``state_dict()``,
    ``parameters()`` and the optimizer. It pickles as its configuration -- its
    public attributes; private ones are runtime state that :meth:`load`
    rebuilds -- and loads the teacher again when unpickled, so checkpoints stay
    small. Being frozen, it is shared rather than copied by ``copy.deepcopy``.

    Subclasses implement :meth:`load` and :meth:`teacher_output`.

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
        student_energy_key: the student's energy output: the teacher's energy
            is its target, and the student's offsets are those of this key.
        student_force_key: the student's force output: the teacher's forces are
            its target, and the student's curvature is taken through it.
    """

    def __init__(
        self,
        teacher_energy_unit: str | float = "eV",
        teacher_distance_unit: str | float = "Ang",
        student_energy_unit: str | float = "eV",
        student_distance_unit: str | float = "Ang",
        cutoff: float | None = None,
        dtype: torch.dtype | str = torch.float32,
        student_energy_key: str = properties.energy,
        student_force_key: str = properties.forces,
    ):
        self.student_energy_key = student_energy_key
        self.student_force_key = student_force_key
        #: the targets this teacher supplies: energy, forces, curvature
        self.target_keys = (
            teacher_key(student_energy_key),
            teacher_key(student_force_key),
            properties.teacher_hvp,
        )
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

    # ------------------------------------------------------------ family hooks

    def load(self) -> Any:
        """Load the teacher. Called at construction and after unpickling."""
        raise NotImplementedError

    def teacher_output(
        self, inputs: dict[str, torch.Tensor], create_graph: bool
    ) -> dict[str, torch.Tensor]:
        """The teacher's ``{"energy": (n_structures,), "forces": (n_atoms, 3)}``
        in teacher units.

        ``inputs`` is a SchNetPack batch in the teacher's units and dtype, with
        the teacher's neighbor list. Its positions are the tensor the forces are
        differentiated by: a translation must build on them, not copy them.
        ``create_graph`` is True when a curvature target is asked for; the
        forces must then come back differentiable with respect to those
        positions, however the family achieves that.
        """
        raise NotImplementedError

    def __call__(
        self, batch: dict[str, torch.Tensor], probe: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Teacher targets for the structures in ``batch``.

        Args:
            batch: student batch, in the student's units.
            probe: ``(n_atoms, 3)`` vector along which to take the teacher's
                Hessian-vector product; None for energy and forces only.
        """
        student_dtype = batch[properties.R].dtype
        inputs = self._teacher_inputs(batch)
        positions = inputs[properties.R]

        create_graph = probe is not None
        hessian_probe = None
        with torch.enable_grad():
            output = self.teacher_output(inputs, create_graph)
            energy = output[properties.energy]
            forces = output[properties.forces]
            if probe is not None:
                hessian_probe = self._hvp(forces, positions, probe)

        energy_key, force_key, hvp_key = self.target_keys
        targets = {
            energy_key: energy.detach().to(torch.float64) * self.energy_conversion,
            force_key: (forces.detach() * self.force_conversion).to(student_dtype),
        }
        if hessian_probe is not None:
            targets[hvp_key] = (hessian_probe * self.hessian_conversion).to(
                student_dtype
            )
        return targets

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

    @staticmethod
    def _hvp(
        forces: torch.Tensor, positions: torch.Tensor, probe: torch.Tensor
    ) -> torch.Tensor:
        if forces.grad_fn is None:
            raise RuntimeError(
                "the teacher's forces are not differentiable with respect to the "
                "positions, so no curvature target can be taken. This happens "
                "when the teacher was exported for inference (frozen, or "
                "optimized for inference), when it predicts forces directly "
                "instead of as -dE/dR, or when its teacher_output() does not "
                "keep the graph of its forces when asked to."
            )
        # F = -dE/dR, so the vector-Jacobian product of F with v is -H v
        (minus_hvp,) = grad(forces, positions, grad_outputs=probe.to(forces.dtype))
        return -minus_hvp

    # ------------------------------------------------------ state and devices

    def _setup(self) -> None:
        self.model = self.load()
        self._device: torch.device | None = None

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
    curvature target is taken so that its ``Forces`` keep their graph.

    Args:
        model_path: path to the saved model.
        model_format: ``"torch"`` for a pickled model (SchNetPack 2.x included),
            ``"torchscript"`` for an archive from ``torch.jit.save``, or
            ``"auto"`` to try TorchScript first.
        teacher_energy_key: the model's energy output.
        teacher_force_key: the model's force output.
        teacher_energy_unit: energy unit the model works in.
        teacher_distance_unit: length unit the model works in.
        student_energy_unit: energy unit of the student.
        student_distance_unit: length unit of the student's batches.
        cutoff: see :class:`TeacherWrapper`.
        dtype: floating dtype the model runs in.
        student_energy_key: see :class:`TeacherWrapper`.
        student_force_key: see :class:`TeacherWrapper`.
    """

    def __init__(
        self,
        model_path: str,
        model_format: str = "auto",
        teacher_energy_key: str = properties.energy,
        teacher_force_key: str = properties.forces,
        teacher_energy_unit: str | float = "eV",
        teacher_distance_unit: str | float = "Ang",
        student_energy_unit: str | float = "eV",
        student_distance_unit: str | float = "Ang",
        cutoff: float | None = None,
        dtype: torch.dtype | str = torch.float32,
        student_energy_key: str = properties.energy,
        student_force_key: str = properties.forces,
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
        super().__init__(
            teacher_energy_unit=teacher_energy_unit,
            teacher_distance_unit=teacher_distance_unit,
            student_energy_unit=student_energy_unit,
            student_distance_unit=student_distance_unit,
            cutoff=cutoff,
            dtype=dtype,
            student_energy_key=student_energy_key,
            student_force_key=student_force_key,
        )

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
        if sensitive:
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
        # its Forces keep their graph only in train mode
        mode = train_mode(self.model) if create_graph else contextlib.nullcontext()
        with mode:
            output = self.model(inputs)
        return {
            properties.energy: output[self.teacher_energy_key],
            properties.forces: output[self.teacher_force_key],
        }


class MaceTeacher(TeacherWrapper):
    """A MACE model as teacher, from a TorchScript archive (e.g. MACE-OFF).

    The archive is made in an environment with ``mace-torch`` installed, so
    SchNetPack itself does not depend on it::

        from e3nn.util import jit
        from mace.calculators import mace_off

        calc = mace_off(model="medium", default_dtype="float64")
        torch.jit.save(jit.compile(calc.models[0]), "MACE-OFF23_medium.pt")

    The teacher works in eV and Å. Its energies are absolute: they include its
    per-element reference energies (E0s). These are summed in float64, read
    from the archive before it is cast to ``dtype``, and added to MACE's
    ``interaction_energy``, so a float32 teacher keeps the digits of its
    energies. Its forces keep their graph when MACE is called with
    ``training=True``, which is passed whenever a curvature target is asked
    for. MACE-OFF is licensed for academic use only (ASL).

    Args:
        model_path: path to the archive written by ``torch.jit.save``.
        teacher_energy_unit: energy unit the model works in.
        teacher_distance_unit: length unit the model works in.
        student_energy_unit: energy unit of the student.
        student_distance_unit: length unit of the student's batches.
        cutoff: see :class:`TeacherWrapper`. Must not be smaller than the
            model's ``r_max`` (5 Å for MACE-OFF23), or pairs the model sees
            would be pruned away.
        dtype: floating dtype the model runs in.
        student_energy_key: see :class:`TeacherWrapper`.
        student_force_key: see :class:`TeacherWrapper`.
    """

    def __init__(
        self,
        model_path: str,
        teacher_energy_unit: str | float = "eV",
        teacher_distance_unit: str | float = "Ang",
        student_energy_unit: str | float = "eV",
        student_distance_unit: str | float = "Ang",
        cutoff: float | None = None,
        dtype: torch.dtype | str = torch.float32,
        student_energy_key: str = properties.energy,
        student_force_key: str = properties.forces,
    ):
        self.model_path = model_path
        super().__init__(
            teacher_energy_unit=teacher_energy_unit,
            teacher_distance_unit=teacher_distance_unit,
            student_energy_unit=student_energy_unit,
            student_distance_unit=student_distance_unit,
            cutoff=cutoff,
            dtype=dtype,
            student_energy_key=student_energy_key,
            student_force_key=student_force_key,
        )

    def load(self) -> nn.Module:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"no teacher model at {self.model_path!r}")
        model = torch.jit.load(self.model_path, map_location="cpu")
        self._e0 = self._reference_energies(model)
        model = model.to(self.dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        r_max = float(model.r_max)
        if self.cutoff is not None and self.cutoff < r_max:
            raise ValueError(
                f"the teacher's cutoff {self.cutoff} is below the model's r_max "
                f"{r_max}; pruning would drop pairs the model sees"
            )
        return model

    @staticmethod
    def _reference_energies(model: nn.Module) -> torch.Tensor | None:
        """The model's E0 table in float64, or None if it has none. Newer MACE
        versions keep one row per head; without a head in its input MACE
        evaluates the first."""
        try:
            table = model.atomic_energies_fn.atomic_energies
        except AttributeError:
            return None
        table = table.detach().to(torch.float64)
        return table[0] if table.dim() == 2 else table

    def teacher_output(
        self, inputs: dict[str, torch.Tensor], create_graph: bool
    ) -> dict[str, torch.Tensor]:
        data = self._mace_input(inputs)
        # MACE keeps its forces' graph through its training argument
        output = self.model(data, training=create_graph)
        return {
            properties.energy: self._energy(output, data),
            properties.forces: output["forces"],
        }

    def _energy(
        self, output: dict[str, torch.Tensor], data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """The E0s summed in float64 plus the interaction energy; MACE's own
        total, summed in its dtype, for an archive without either."""
        interaction = output.get("interaction_energy")
        if self._e0 is None or interaction is None:
            warnings.warn(
                "the MACE archive provides no E0 table or no interaction_energy, "
                "so its energies are summed in its own dtype and may lose digits "
                "in float32",
                stacklevel=2,
            )
            return output["energy"]
        e0 = self._e0.to(interaction.device)[data["node_attrs"].argmax(-1)]
        energy = torch.zeros(
            interaction.shape[0], dtype=torch.float64, device=interaction.device
        )
        return energy.index_add(0, data["batch"], e0) + interaction.to(torch.float64)

    def _mace_input(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The dict MACE reads, built on the batch's positions and pruned list."""
        atomic_numbers = inputs[properties.Z]
        # MACE's one-hot columns follow the order of its own element table
        node_attrs = atomic_numbers.unsqueeze(-1) == self.model.atomic_numbers
        uncovered = ~node_attrs.any(-1)
        if uncovered.any():
            raise ValueError(
                "the teacher does not cover the atomic numbers "
                f"{atomic_numbers[uncovered].unique().tolist()}"
            )
        positions = inputs[properties.R]
        n_atoms = inputs[properties.n_atoms]
        cell = inputs.get(properties.cell)
        data = {
            "positions": positions,
            "node_attrs": node_attrs.to(positions.dtype),
            # MACE's edge vectors are positions[edge_index[1]] -
            # positions[edge_index[0]] + shifts, SchNetPack's are R[idx_j] -
            # R[idx_i] + offsets
            "edge_index": torch.stack(
                (inputs[properties.idx_i], inputs[properties.idx_j])
            ),
            "shifts": inputs[properties.offsets],
            "batch": inputs[properties.idx_m],
            "ptr": torch.cat((n_atoms.new_zeros(1), n_atoms.cumsum(0))),
            "cell": (
                positions.new_zeros(3 * n_atoms.shape[0], 3)
                if cell is None
                else cell.reshape(-1, 3)
            ),
        }
        return data

"""MACE as a SchNetPack model.

:class:`MaceModel` puts a MACE model -- the *core* -- behind SchNetPack's batch
format, so that one object serves as the potential of a
:mod:`~schnetpack.dynamics` driver and as distillation teacher
(:class:`~schnetpack.train.MaceTeacher`). The core is a ``torch.export``
archive (``.pt2``), read by :func:`load_mace` into an :class:`ExportedMace`, as
:func:`~schnetpack.interfaces.mace_interface.export_mace_off` writes them;
loading it needs no ``mace-torch``.

MACE works in eV and Å, the defaults of
:class:`~schnetpack.dynamics.ForceCalculator`. Its neighbor list must cover the
core's :attr:`~MaceModel.r_max`::

    model = MaceModel(load_mace(path), dtype="float32")
    calculator = ForceCalculator(
        model,
        neighbor_list=BatchNeighborList(MatScipyNeighborList(cutoff=model.r_max)),
    )

Cast with ``MaceModel(dtype=...)`` rather than ``ForceCalculator(dtype=...)``:
the latter casts the whole model, and rounds the float64 E0s with it.

A ``MaceModel`` trains like any SchNetPack model (``experiment=mace_finetune``),
also from random weights (``experiment=mace_scratch``, on a core that
:func:`~schnetpack.interfaces.mace_interface.build_mace` writes), and pickles
with ``torch.save``, the core as its embedded archive.
Training compares MACE's interaction energy with labels from which the data
pipeline removed the same E0s the postprocessors add: see ``e0s`` and
:func:`mace_atomrefs`.
"""

import hashlib
import io
import json
import os
import warnings
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.model.base import AtomisticModel
from schnetpack.transform import AddOffsets, CastTo64, Transform
from schnetpack.utils import as_dtype

__all__ = ["MaceModel", "ExportedMace", "load_mace", "mace_atomrefs"]

#: where the E0s come from: the core's own table, the dataset's atomrefs, or
#: atomrefs estimated on the training split
E0_SOURCES = ("model", "dataset", "estimated")

#: the arguments of an exported MACE energy, in order: the keys of MACE's
#: input dict
MACE_INPUTS = (
    "positions",
    "node_attrs",
    "edge_index",
    "shifts",
    "batch",
    "ptr",
    "cell",
)

#: name of the extra file in which an exported archive keeps what MACE's
#: interface reads off the model: elements, cutoff, E0s and energy shift
METADATA = "mace.json"


class MaceModel(AtomisticModel):
    """A MACE model on SchNetPack batches.

    The batch must hold a neighbor list (``idx_i``, ``idx_j``, ``offsets``)
    that covers the core's ``r_max``; pairs beyond it are harmless, as MACE's
    radial envelope vanishes there. Positions are in Å and energies in eV, the
    units MACE models work in.

    The raw energy is MACE's ``interaction_energy``, plus a per-element E0
    correction (zero unless fitted). The per-element reference energies (E0s)
    are SchNetPack atomrefs, added by the postprocessors after ``CastTo64`` --
    in float64, so that absolute energies keep their digits. By default the
    postprocessors take them from the core's own E0 table.

    For training on data of another level of theory, ``e0s="dataset"`` takes
    the dataset's atomrefs (e.g. isolated-atom energies) instead. Without
    any, ``fit_e0_correction`` fits a correction per element by least squares
    on what the pretrained model misses over the training split, as MACE's
    ``E0s="estimated"`` does; on data of one composition only its sum over
    the molecule is determined, and the minimum-norm solution is taken.

    A core with random weights (:func:`~schnetpack.interfaces.mace_interface.build_mace`)
    has no E0s of its own: ``e0s="estimated"`` fits them per element to the
    energies of the training split, as MACE's ``E0s="average"`` does. The
    pipeline's ``RemoveOffsets`` needs ``estimate_atomref=True`` to remove the
    same ones. Atomrefs stored with the dataset take precedence.

    The core's forces keep their graph in train mode, through MACE's
    ``training`` argument. Only the first head of a multi-head core is used.

    Args:
        core: the MACE model, as :func:`load_mace` returns it.
        energy_key: output key of the energy.
        force_key: output key of the forces.
        postprocessors: transforms applied to the outputs when
            ``do_postprocessing`` is on. None adds the E0s ``e0s`` names.
        do_postprocessing: whether the postprocessors run.
        dtype: floating dtype the core is cast to; None keeps its own. The
            default E0s are read before the cast, so they stay exact.
        e0s: ``"model"`` for the core's E0 table, ``"dataset"`` for the
            atomrefs of the training data, ``"estimated"`` for atomrefs
            estimated on the training split; the latter two are taken at
            :meth:`initialize_transforms`. Only read without
            ``postprocessors``.
        fit_e0_correction: fit the E0 correction at
            :meth:`initialize_transforms`.
    """

    def __init__(
        self,
        core: nn.Module,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        postprocessors: Sequence[Transform] | None = None,
        do_postprocessing: bool = True,
        dtype: torch.dtype | str | None = None,
        e0s: str = "model",
        fit_e0_correction: bool = False,
    ):
        _check_e0s(e0s)
        if postprocessors is None:
            atomrefs = reference_energies(core) if e0s == "model" else None
            postprocessors = [
                CastTo64(),
                AddOffsets(
                    energy_key,
                    add_atomrefs=True,
                    atomrefs=atomrefs,
                    estimate_atomref=e0s == "estimated",
                ),
            ]
        if dtype is not None:
            core = core.to(as_dtype(dtype))
        super().__init__(
            postprocessors=list(postprocessors), do_postprocessing=do_postprocessing
        )
        self.core = core
        self.energy_key = energy_key
        self.force_key = force_key
        self.e0s = e0s
        self.fit_e0_correction = fit_e0_correction
        self._e0_correction_fitted = False
        self.required_derivatives = [properties.R]
        self.model_outputs = [energy_key, force_key]
        size = int(core.atomic_numbers.max()) + 1
        self.register_buffer("e0_correction", torch.zeros(size, dtype=torch.float64))

    @property
    def r_max(self) -> float:
        """The core's cutoff, in Å."""
        return float(self.core.r_max)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = self.initialize_derivatives(inputs)
        data = mace_input(inputs, self.core.atomic_numbers)
        output = self.core(data, training=self.training)

        interaction = output.get("interaction_energy")
        if interaction is None:
            raise KeyError(
                "the MACE model returns no interaction_energy, which MaceModel "
                "takes as its energy before the E0s are added"
            )
        correction = self.e0_correction.to(interaction.dtype)[inputs[properties.Z]]
        inputs[self.energy_key] = interaction.index_add(
            0, inputs[properties.idx_m], correction
        )
        inputs[self.force_key] = output["forces"]

        inputs = self.postprocess(inputs)
        return self.extract_outputs(inputs)

    def initialize_transforms(self, datamodule):
        """Initialize the postprocessors from ``datamodule``, and fit the E0
        correction if asked to. The fit is stored with the datamodule's
        statistics, keyed by the core's weights, so a resumed run reads it
        back."""
        super().initialize_transforms(datamodule)
        if not self.fit_e0_correction or self._e0_correction_fitted:
            return
        key = (
            f"mace_e0_correction:{self.energy_key}:{self.e0s}:{_fingerprint(self.core)}"
        )
        correction = datamodule.provider.read_or_compute(
            key, lambda: self._fit_e0_correction(datamodule.train_dataloader())
        )
        self.e0_correction = torch.as_tensor(correction, dtype=torch.float64).to(
            self.e0_correction.device
        )
        self._e0_correction_fitted = True

    def _fit_e0_correction(self, loader) -> np.ndarray:
        """The E0 correction by which the model best meets the energies of
        ``loader``, offsets removed, as least-squares fit on the element
        counts."""
        numbers = self.core.atomic_numbers.long()
        device = self.e0_correction.device
        training, postprocessing = self.training, self.do_postprocessing
        self.eval()
        self.do_postprocessing = False
        counts, residuals = [], []
        try:
            with torch.enable_grad():
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    # read first: the model writes its energy into the batch
                    target = batch[self.energy_key].detach().double()
                    energy = self(batch)[self.energy_key].detach().double()
                    columns = batch[properties.Z].unsqueeze(-1) == numbers
                    counts.append(
                        torch.zeros(
                            len(target),
                            len(numbers),
                            dtype=torch.float64,
                            device=device,
                        ).index_add(0, batch[properties.idx_m], columns.double())
                    )
                    residuals.append(target - energy)
        finally:
            self.train(training)
            self.do_postprocessing = postprocessing
        solution = torch.linalg.lstsq(
            torch.cat(counts).cpu(),
            torch.cat(residuals).cpu().unsqueeze(-1),
            driver="gelsd",
        ).solution.squeeze(-1)
        correction = self.e0_correction.detach().cpu().clone()
        correction[numbers.cpu()] += solution
        return correction.numpy()


class ExportedMace(nn.Module):
    """A MACE model exported with ``torch.export``, behind the interface
    :class:`MaceModel` reads off a MACE model: ``forward(data, training)``
    returning ``interaction_energy`` and ``forces``, and ``atomic_numbers``,
    ``r_max``, the E0 table ``atomic_energies`` and the energy ``shift``.

    The archive holds the interaction energy as a function of MACE's inputs
    (:data:`MACE_INPUTS`); the forces are its position derivative, taken
    here, with their graph kept in train mode. Device and floating dtype are
    written into the exported graph; moving or casting the module rewrites
    them. It pickles as its archive and current weights, and is restored on
    the CPU.

    Args:
        archive: the bytes of an archive as
            :func:`~schnetpack.interfaces.mace_interface.export_mace` writes it.
    """

    def __init__(self, archive: bytes):
        super().__init__()
        extra_files = {METADATA: ""}
        program = torch.export.load(io.BytesIO(archive), extra_files=extra_files)
        metadata = json.loads(extra_files[METADATA])
        self._archive = archive
        self.module = program.module()
        self.register_buffer("atomic_numbers", torch.tensor(metadata["atomic_numbers"]))
        self.register_buffer(
            "r_max", torch.tensor(metadata["r_max"], dtype=torch.float64)
        )
        self.register_buffer(
            "atomic_energies",
            torch.tensor(metadata["atomic_energies"], dtype=torch.float64),
        )
        self.register_buffer(
            "shift", torch.tensor(metadata["shift"], dtype=torch.float64)
        )
        self._graph_target = self._target()

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        positions = data["positions"]
        if compute_force and not positions.requires_grad:
            positions.requires_grad_(True)
        energy = self.module(*(data[name] for name in MACE_INPUTS))
        forces = None
        if compute_force:
            (gradient,) = grad(
                [energy.sum()],
                [positions],
                create_graph=training,
                retain_graph=training,
            )
            forces = -gradient
        return {"interaction_energy": energy, "forces": forces}

    def train(self, mode: bool = True) -> "ExportedMace":
        # an exported graph refuses train()/eval(); it holds no layer whose
        # forward depends on the mode
        self.training = mode
        return self

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        target = self._target()
        if target != self._graph_target:
            _retarget(self.module, *target)
            self._graph_target = target
        return self

    def _target(self) -> tuple[torch.device, torch.dtype]:
        tensors = [*self.module.parameters(), *self.module.buffers()]
        reference = next(t for t in tensors if t.is_floating_point())
        return reference.device, reference.dtype

    def __reduce__(self):
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        requires_grad = [p.requires_grad for p in self.parameters()]
        return (
            _restore_exported,
            (self._archive, state, self._target()[1], requires_grad, self.training),
        )

    def __deepcopy__(self, memo):
        restore, args = self.__reduce__()
        return restore(*args).to(self._target()[0])


def _restore_exported(archive, state, dtype, requires_grad, training) -> ExportedMace:
    core = ExportedMace(archive).to(dtype)
    core.load_state_dict(state)
    for parameter, flag in zip(core.parameters(), requires_grad):
        parameter.requires_grad_(flag)
    return core.train(training)


def _retarget(module: torch.fx.GraphModule, device, dtype) -> None:
    """Rewrite the devices and floating dtypes written into ``module``'s
    graph, as ``torch.export`` does not let ``.to`` reach them."""

    def swap(value):
        if isinstance(value, torch.device):
            return device
        if isinstance(value, torch.dtype) and value.is_floating_point:
            return dtype
        return value

    for node in module.graph.nodes:
        node.args = tuple(swap(value) for value in node.args)
        node.kwargs = {key: swap(value) for key, value in node.kwargs.items()}
    module.recompile()


def mace_input(
    inputs: dict[str, torch.Tensor], atomic_numbers: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The dict MACE reads, built on the batch's positions and neighbor list.

    Args:
        inputs: SchNetPack batch with a neighbor list.
        atomic_numbers: the core's element table; MACE's one-hot columns follow
            its order.
    """
    numbers = inputs[properties.Z]
    node_attrs = numbers.unsqueeze(-1) == atomic_numbers
    uncovered = ~node_attrs.any(-1)
    if uncovered.any():
        raise ValueError(
            "the MACE model does not cover the atomic numbers "
            f"{numbers[uncovered].unique().tolist()}"
        )
    positions = inputs[properties.R]
    n_atoms = inputs[properties.n_atoms]
    cell = inputs.get(properties.cell)
    return {
        "positions": positions,
        "node_attrs": node_attrs.to(positions.dtype),
        "edge_index": torch.stack((inputs[properties.idx_i], inputs[properties.idx_j])),
        "shifts": inputs[properties.offsets],
        "batch": inputs[properties.idx_m],
        "ptr": torch.cat((n_atoms.new_zeros(1), n_atoms.cumsum(0))),
        "cell": (
            positions.new_zeros(3 * n_atoms.shape[0], 3)
            if cell is None
            else cell.reshape(-1, 3)
        ),
    }


def reference_energies(core: nn.Module) -> torch.Tensor:
    """The core's E0s in float64, indexed by atomic number, as atomrefs for
    :class:`~schnetpack.transform.AddOffsets`. Newer MACE versions keep one
    row per head; the first one is taken."""
    table = _e0_table(core).detach().to(torch.float64)
    if table.dim() == 2:
        if table.shape[0] > 1:
            warnings.warn(
                f"the MACE model has {table.shape[0]} heads; only its first is used",
                stacklevel=3,
            )
        table = table[0]
    numbers = core.atomic_numbers.long()
    atomrefs = torch.zeros(int(numbers.max()) + 1, dtype=torch.float64)
    atomrefs[numbers] = table
    return atomrefs


def energy_shift(core: nn.Module) -> float:
    """The energy the core's ``ScaleShiftBlock`` adds to every atom, of its
    first head; 0 for a core without one."""
    if isinstance(core, ExportedMace):
        return float(core.shift.detach().reshape(-1)[0])
    block = getattr(core, "scale_shift", None)
    if block is None:
        return 0.0
    return float(block.shift.detach().reshape(-1)[0])


def _e0_table(core: nn.Module) -> torch.Tensor:
    if isinstance(core, ExportedMace):
        return core.atomic_energies
    try:
        return core.atomic_energies_fn.atomic_energies
    except AttributeError:
        raise AttributeError(
            "the MACE model has no E0 table (atomic_energies_fn.atomic_energies); "
            "pass postprocessors to MaceModel to supply its reference energies"
        ) from None


def _fingerprint(core: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in core.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _check_e0s(e0s: str) -> None:
    if e0s not in E0_SOURCES:
        raise ValueError(f"unknown E0 source {e0s!r}, expected one of {E0_SOURCES}")


def load_mace(path: str) -> nn.Module:
    """Load a MACE model, to be wrapped in :class:`MaceModel`. Needs no
    ``mace-torch``.

    The model comes back trainable, on the CPU.

    Args:
        path: a ``torch.export`` archive as
            :func:`~schnetpack.interfaces.mace_interface.export_mace` writes
            it (e.g. :func:`~schnetpack.interfaces.mace_interface.export_mace_off`),
            loaded into an :class:`ExportedMace`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no MACE model at {path!r}")
    with open(path, "rb") as file:
        core = ExportedMace(file.read())
    # MACE's calculator freezes the parameters, so models made from it store
    # them frozen
    return core.requires_grad_(True)


def mace_atomrefs(path: str, e0s: str = "model") -> torch.Tensor | None:
    """The atomrefs the data pipeline removes from the energy labels for
    training a :class:`MaceModel` with the same ``e0s``: the archive's E0s,
    or None for ``"dataset"`` and ``"estimated"``, so that ``RemoveOffsets``
    reads the atomrefs of the training data as the model's ``AddOffsets`` does
    (for ``"estimated"``, with ``estimate_atomref=True``).

    Args:
        path: the MACE archive (see :func:`load_mace`).
        e0s: one of :data:`E0_SOURCES`, as for :class:`MaceModel`.
    """
    _check_e0s(e0s)
    return reference_energies(load_mace(path)) if e0s == "model" else None

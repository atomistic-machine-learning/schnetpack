"""MACE as a SchNetPack model.

:class:`MaceModel` puts a MACE model -- the *core* -- behind SchNetPack's batch
format, so that one object serves as the potential of a
:mod:`~schnetpack.dynamics` driver and as distillation teacher
(:class:`~schnetpack.train.MaceTeacher`). The core is a TorchScript archive,
read by :func:`load_mace`, as
:func:`~schnetpack.interfaces.mace_interface.compile_mace_off` writes them;
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
"""

import os
import warnings
from collections.abc import Sequence

import torch
from torch import nn

from schnetpack import properties
from schnetpack.model.base import AtomisticModel
from schnetpack.transform import AddOffsets, CastTo64, Transform
from schnetpack.utils import as_dtype

__all__ = ["MaceModel", "load_mace"]


class MaceModel(AtomisticModel):
    """A MACE model on SchNetPack batches.

    The batch must hold a neighbor list (``idx_i``, ``idx_j``, ``offsets``)
    that covers the core's ``r_max``; pairs beyond it are harmless, as MACE's
    radial envelope vanishes there. Positions are in Å and energies in eV, the
    units MACE models work in.

    The raw energy is MACE's ``interaction_energy``. The per-element reference
    energies (E0s) are SchNetPack atomrefs, added by the postprocessors after
    ``CastTo64`` -- in float64, so that absolute energies keep their digits.
    By default the postprocessors take them from the core's own E0 table.

    The core's forces keep their graph in train mode, through MACE's
    ``training`` argument. Only the first head of a multi-head core is used.

    Args:
        core: the MACE model, a TorchScript archive (see :func:`load_mace`).
        energy_key: output key of the energy.
        force_key: output key of the forces.
        postprocessors: transforms applied to the outputs when
            ``do_postprocessing`` is on. None adds the core's E0s.
        do_postprocessing: whether the postprocessors run.
        dtype: floating dtype the core is cast to; None keeps its own. The
            default E0s are read before the cast, so they stay exact.
    """

    def __init__(
        self,
        core: nn.Module,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        postprocessors: Sequence[Transform] | None = None,
        do_postprocessing: bool = True,
        dtype: torch.dtype | str | None = None,
    ):
        if postprocessors is None:
            postprocessors = [
                CastTo64(),
                AddOffsets(
                    energy_key, add_atomrefs=True, atomrefs=reference_energies(core)
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
        self.required_derivatives = [properties.R]
        self.model_outputs = [energy_key, force_key]

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
        inputs[self.energy_key] = interaction
        inputs[self.force_key] = output["forces"]

        inputs = self.postprocess(inputs)
        return self.extract_outputs(inputs)


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


def _e0_table(core: nn.Module) -> torch.Tensor:
    try:
        return core.atomic_energies_fn.atomic_energies
    except AttributeError:
        raise AttributeError(
            "the MACE model has no E0 table (atomic_energies_fn.atomic_energies); "
            "pass postprocessors to MaceModel to supply its reference energies"
        ) from None


def load_mace(path: str) -> nn.Module:
    """Load a MACE model from a TorchScript archive, to be wrapped in
    :class:`MaceModel`. Needs no ``mace-torch``.

    Args:
        path: the archive, as ``torch.jit.save`` writes it (e.g.
            :func:`~schnetpack.interfaces.mace_interface.compile_mace_off`).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no MACE model at {path!r}")
    return torch.jit.load(path, map_location="cpu")

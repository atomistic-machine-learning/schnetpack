"""MACE models exported with ``torch.export``, for
:class:`~schnetpack.model.MaceModel` and :class:`~schnetpack.train.MaceTeacher`.

:func:`export_mace` exports a MACE model -- any module with MACE's interface --
to an archive that :func:`~schnetpack.model.load_mace` reads without
``mace-torch``. :func:`export_mace_off` builds MACE-OFF23 and writes
``MACE-OFF23_<size>_mace-<version>.pt2`` to :data:`CACHE_DIR`;
:func:`build_mace` writes a MACE model with random weights, to be trained from
scratch (``experiment=mace_scratch``). Both are developer tools and need
``mace-torch``, which SchNetPack does not depend on::

    pip install "schnetpack[mace]"
"""

import contextlib
import io
import json
import os
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.export import Dim

from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.model.mace import MACE_INPUTS, METADATA, ExportedMace
from schnetpack.transform import MatScipyNeighborList

__all__ = [
    "CACHE_DIR",
    "MACE_OFF_SIZES",
    "build_mace",
    "export_mace",
    "export_mace_off",
]

#: where :func:`export_mace_off` writes by default; ignored by git
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MACE_OFF_SIZES = ("small", "medium", "large")

#: set by ``import mace`` so that e3nn 0.4.4 and MACE's pickled models load on
#: torch >= 2.6; left set, it would turn off ``weights_only`` for every later
#: ``torch.load`` in the process
_NO_WEIGHTS_ONLY = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"

#: the dimension of each input that varies with the batch: atoms, edges or
#: structures
_DYNAMIC = {
    "positions": 0,
    "node_attrs": 0,
    "edge_index": 1,
    "shifts": 0,
    "batch": 0,
    "ptr": 0,
    "cell": 0,
}


def export_mace(model: nn.Module, path: str) -> str:
    """Export ``model``'s interaction energy with ``torch.export`` and write
    the archive to ``path``, to be read by
    :func:`~schnetpack.model.load_mace`.

    ``model`` has MACE's interface: ``forward(data, training, compute_force)``
    returning ``interaction_energy``, ``atomic_numbers``, ``r_max`` and
    ``atomic_energies_fn.atomic_energies``, and optionally ``scale_shift``. It
    must hold no TorchScript, as MACE's e3nn tensor products do unless built
    with e3nn's code generation off (see :func:`export_mace_off`). The export
    fails if the model branches on the number of atoms, edges or structures,
    and is checked to match ``model`` on batches of other sizes.

    Args:
        model: the MACE model, in the dtype the archive is to have.
        path: where the archive is written.
    """
    energy = _Energy(model)
    example = _example(model, n_structures=2, seed=0)
    # recent torch (2.14, not 2.7) guards on whether a dynamic size is 1, e.g.
    # in the contiguity check of index_add's decomposition, which would bound
    # the export to batches of at least 2 structures; size-oblivious tracing
    # assumes such a size is not 1 instead, and the checks below hold the
    # export to a single structure too
    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
        program = torch.export.export(
            energy,
            tuple(example[name] for name in MACE_INPUTS),
            dynamic_shapes=tuple({_DYNAMIC[name]: Dim.DYNAMIC} for name in MACE_INPUTS),
            strict=True,
        )

    table = model.atomic_energies_fn.atomic_energies
    block = getattr(model, "scale_shift", None)
    metadata = {
        "atomic_numbers": model.atomic_numbers.tolist(),
        "r_max": float(model.r_max),
        "atomic_energies": table.detach().double().tolist(),
        "shift": [0.0] if block is None else block.shift.reshape(-1).tolist(),
    }
    archive = io.BytesIO()
    torch.export.save(program, archive, extra_files={METADATA: json.dumps(metadata)})

    exported = ExportedMace(archive.getvalue())
    for n_structures, seed in ((1, 1), (3, 2)):
        data = _example(model, n_structures, seed)
        torch.testing.assert_close(
            exported(dict(data), compute_force=False)["interaction_energy"],
            energy(*(data[name] for name in MACE_INPUTS)),
            msg=f"the export differs from the model on {n_structures} structures",
        )

    with open(path, "wb") as file:
        file.write(archive.getvalue())
    return path


class _Energy(nn.Module):
    """MACE's interaction energy as a function of its inputs; the forces are
    taken outside the exported graph, as ``torch.export`` captures no
    ``autograd.grad``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, positions, node_attrs, edge_index, shifts, batch, ptr, cell):
        data = {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "batch": batch,
            "ptr": ptr,
            "cell": cell,
        }
        return self.model(data, training=False, compute_force=False)[
            "interaction_energy"
        ]


def _example(model: nn.Module, n_structures: int, seed: int) -> dict:
    """MACE's inputs for ``n_structures`` small molecules of ``model``'s
    elements (3, 4, ... atoms), every pair of atoms in each an edge."""
    generator = torch.Generator().manual_seed(seed)
    dtype = next(
        t for t in [*model.parameters(), *model.buffers()] if t.is_floating_point()
    ).dtype
    n_elements = len(model.atomic_numbers)
    sizes = [3 + index for index in range(n_structures)]
    positions, columns, edges, batch = [], [], [], []
    start = 0
    for index, size in enumerate(sizes):
        # a chain, 1.2 Å between neighbors, jittered off the line
        chain = torch.arange(size, dtype=torch.float64)[:, None] * torch.tensor(
            [1.2, 0.0, 0.0], dtype=torch.float64
        )
        positions.append(chain + 0.3 * torch.rand(size, 3, generator=generator))
        columns.append((torch.arange(size) + index) % n_elements)
        pairs = torch.cartesian_prod(torch.arange(size), torch.arange(size))
        edges.append(pairs[pairs[:, 0] != pairs[:, 1]] + start)
        batch.append(torch.full((size,), index))
        start += size
    edge_index = torch.cat(edges).T.contiguous()
    return {
        "positions": torch.cat(positions).to(dtype),
        "node_attrs": nn.functional.one_hot(torch.cat(columns), n_elements).to(dtype),
        "edge_index": edge_index,
        "shifts": torch.zeros(edge_index.shape[1], 3, dtype=dtype),
        "batch": torch.cat(batch),
        "ptr": torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()]),
        "cell": torch.zeros(3 * n_structures, 3, dtype=dtype),
    }


def export_mace_off(size: str = "medium", cache_dir: str = CACHE_DIR) -> str:
    """Export MACE-OFF23 (see :func:`export_mace`) and return the archive's
    path.

    MACE downloads the weights on first use. The model is exported in
    float64; an archive already in ``cache_dir`` is reused.

    Args:
        size: ``"small"``, ``"medium"`` or ``"large"``.
        cache_dir: directory the archive is written to.
    """
    if size not in MACE_OFF_SIZES:
        raise ValueError(
            f"unknown MACE-OFF size {size!r}, expected one of {MACE_OFF_SIZES}"
        )

    with _mace_environment():
        mace = _import_mace()
        path = os.path.join(cache_dir, f"MACE-OFF23_{size}_mace-{mace.__version__}.pt2")
        if not os.path.exists(path):
            os.makedirs(cache_dir, exist_ok=True)
            # written under another name first, so that an interrupted run is
            # not taken for a cached archive
            partial = path + ".part"
            export_mace(_mace_off(size), partial)
            os.replace(partial, path)
    return path


def _mace_off(size: str) -> nn.Module:
    """MACE-OFF23 in float64, rebuilt from its config with e3nn's code
    generation off: the pickled model holds TorchScript, which
    ``torch.export`` cannot trace."""
    from mace.calculators import mace_off
    from mace.tools.scripts_utils import extract_config_mace_model

    pickled = mace_off(model=size, default_dtype="float64", device="cpu").models[0]
    model = _traceable(type(pickled), extract_config_mace_model(pickled))
    missing, unexpected = model.load_state_dict(pickled.state_dict(), strict=False)
    # buffers of newer MACE versions than the one that pickled the model keep
    # the values they are built with
    if unexpected or any(not key.endswith("_zeroed") for key in missing):
        raise RuntimeError(
            f"MACE-OFF23 {size} does not fit its rebuilt model: missing {missing}, "
            f"unexpected {unexpected}"
        )
    return model.eval()


def build_mace(
    path: str,
    dataset: ASEAtomsData,
    indices: Sequence[int] | None = None,
    *,
    forces_key: str = properties.forces,
    r_max: float = 5.0,
    num_channels: int = 128,
    max_L: int = 1,
    correlation: int = 3,
    num_interactions: int = 2,
    max_ell: int = 3,
    num_bessel: int = 8,
    num_polynomial_cutoff: int = 5,
    radial_MLP: Sequence[int] = (64, 64, 64),
    MLP_irreps: str = "16x0e",
    scale: float | None = None,
    atomic_numbers: Sequence[int] | None = None,
    seed: int = 0,
) -> str:
    """Build a MACE model with random weights for ``dataset``, export it (see
    :func:`export_mace`) to ``path`` in float64 and return the path.

    The model is MACE's default ``ScaleShiftMACE``, and the keywords and their
    defaults are those of MACE's training script. What MACE reads off the
    training data when it builds a model is taken from ``dataset``:

    - the elements, from all of ``dataset``, so that no validation or test
      structure has an element the model does not cover, unless
      ``atomic_numbers`` is given;
    - the average number of neighbors within ``r_max``, the normalization of
      MACE's message passing, from ``indices``;
    - the scale of the atomic energies, as the RMS of the force components
      over ``indices`` (MACE's ``rms_forces_scaling``), unless ``scale`` is
      given.

    The scale comes from force labels, so ``indices`` should be the
    training split. Positions must be in Å and forces in eV/Å: open
    ``dataset`` in these units. The model's E0s and energy shift are zero:
    train it with ``e0s="estimated"`` (see :class:`~schnetpack.model.MaceModel`).
    Any per-atom energy shift goes into the E0s, which are fitted per element.

    ``atomic_numbers`` gives the model elements beyond those of ``dataset``,
    e.g. those of a pretrained model to compare with: each element has weights
    of its own, so the model has as many parameters as one built for these
    elements. The weights of elements ``dataset`` lacks get no gradient.

    Args:
        path: where the archive is written.
        dataset: the data the model is trained on.
        indices: the structures of ``dataset`` the statistics are taken over;
            None for all.
        forces_key: the property of ``dataset`` that holds the forces.
        r_max: cutoff radius, in Å.
        num_channels: number of channels of the hidden features.
        max_L: highest symmetry order of the hidden features.
        correlation: correlation order of each interaction.
        num_interactions: number of interactions.
        max_ell: highest degree of the spherical harmonics.
        num_bessel: number of radial Bessel functions.
        num_polynomial_cutoff: order of the polynomial cutoff.
        radial_MLP: hidden layer widths of the radial MLP.
        MLP_irreps: hidden irreps of the readout MLP.
        scale: scale of the atomic energies; None for the RMS of the forces.
        atomic_numbers: the elements of the model, which must cover those of
            ``dataset``; None for those of ``dataset``.
        seed: seed of the random weights.
    """
    with _mace_environment():
        _import_mace()
        from e3nn import o3

        from mace import modules

        numbers = _elements(dataset)
        if atomic_numbers is not None:
            missing = sorted(set(numbers) - set(atomic_numbers))
            if missing:
                raise ValueError(
                    f"atomic_numbers do not cover the elements {missing} of the dataset"
                )
            numbers = sorted(set(atomic_numbers))
        if scale is None:
            scale = _force_rms(dataset, indices, forces_key)
        config = {
            "r_max": r_max,
            "num_bessel": num_bessel,
            "num_polynomial_cutoff": num_polynomial_cutoff,
            "max_ell": max_ell,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            # MACE's training script swaps the residual block out of the first
            # layer
            "interaction_cls_first": modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            "num_interactions": num_interactions,
            "num_elements": len(numbers),
            "hidden_irreps": o3.Irreps(
                (num_channels * o3.Irreps.spherical_harmonics(max_L))
                .sort()
                .irreps.simplify()
            ),
            "MLP_irreps": o3.Irreps(MLP_irreps),
            "atomic_energies": np.zeros(len(numbers)),
            "avg_num_neighbors": _avg_num_neighbors(dataset, indices, r_max),
            "atomic_numbers": numbers,
            "correlation": correlation,
            "gate": modules.gate_dict["silu"],
            "radial_MLP": list(radial_MLP),
            "atomic_inter_scale": scale,
            "atomic_inter_shift": 0.0,
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _traceable(modules.ScaleShiftMACE, config)
        return export_mace(model.eval(), path)


def _elements(dataset: ASEAtomsData) -> list[int]:
    """The atomic numbers in ``dataset``, in ascending order."""
    numbers = set()
    for structure in dataset.iter_properties(load_properties=[]):
        numbers.update(structure[properties.Z].tolist())
    return sorted(numbers)


def _avg_num_neighbors(
    dataset: ASEAtomsData, indices: Sequence[int] | None, r_max: float
) -> float:
    """The mean number of neighbors within ``r_max`` of the atoms that have
    any, as MACE counts them."""
    neighbor_list = MatScipyNeighborList(cutoff=r_max)
    counts = []
    for structure in dataset.iter_properties(indices, load_properties=[]):
        idx_i = neighbor_list(structure)[properties.idx_i]
        counts.append(torch.unique(idx_i, return_counts=True)[1])
    return torch.cat(counts).double().mean().item()


def _force_rms(
    dataset: ASEAtomsData, indices: Sequence[int] | None, forces_key: str
) -> float:
    """The RMS of the force components in ``dataset``."""
    squares = [
        structure[forces_key].double().square().reshape(-1)
        for structure in dataset.iter_properties(
            indices, load_properties=[forces_key], load_structure=False
        )
    ]
    return torch.cat(squares).mean().sqrt().item()


def _traceable(cls: type, config: dict) -> nn.Module:
    """``cls(**config)`` in float64, built with e3nn's code generation off: it
    would hold TorchScript, which ``torch.export`` cannot trace. The constants
    of the tensor products are computed in float64 too."""
    from mace.tools.compile import disable_e3nn_codegen, simplify

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with disable_e3nn_codegen():
            model = cls(**config)
    finally:
        torch.set_default_dtype(default)
    return simplify(model).to(torch.float64)


def _import_mace():
    try:
        import mace
    except ImportError as err:
        raise ImportError(
            "mace-torch is needed to export a MACE model; install it with "
            "pip install 'schnetpack[mace]'"
        ) from err
    return mace


@contextlib.contextmanager
def _mace_environment():
    """Unsafe ``torch.load`` while MACE loads its models, undone afterwards:
    ``import mace`` sets it, but on its first import only."""
    previous = os.environ.get(_NO_WEIGHTS_ONLY)
    os.environ[_NO_WEIGHTS_ONLY] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_NO_WEIGHTS_ONLY, None)
        else:
            os.environ[_NO_WEIGHTS_ONLY] = previous

"""SchNetPack potentials compiled ahead of time for LAMMPS.

:func:`deploy_lammps` compiles :class:`LammpsPotential` -- a model's energy, its
per-atom contributions, the forces and the virial as functions of flat tensors -- with
AOTInductor into a package that ``pair_style schnetpack`` (``interfaces/lammps``) loads
without Python. The graph is recorded by the tracer of
:class:`~schnetpack.model.CompiledPotential`, so the derivatives are part of it, and
exported with ``torch.export``. A package runs only with the torch version it was built
with and on the device (and GPU architecture) it was built for, so it is built on the
machine LAMMPS runs on, with ``spkdeploy``.
"""

import copy
import os
import tempfile

import torch
from torch import nn
from torch.export import Dim
from torch.func import functional_call

import schnetpack
from schnetpack import properties
from schnetpack.model.atomistic import Atomwise, Forces, PairwiseDistances, Strain
from schnetpack.model.base import NeuralNetworkPotential
from schnetpack.model.compile import trace
from schnetpack.model.representation import PaiNN, SchNet, SO3net
from schnetpack.transform import AddOffsets, CastTo32, CastTo64

__all__ = ["LAMMPS_INPUTS", "LAMMPS_OUTPUTS", "LammpsPotential", "deploy_lammps"]

#: the arguments of a deployed potential, in order
LAMMPS_INPUTS = ("positions", "atomic_numbers", "idx_i", "idx_j", "offsets", "local")
#: its outputs, in order
LAMMPS_OUTPUTS = ("energy", "energy_atom", "forces", "virial")

#: bounds of the dynamic sizes; unbounded ones make the export guard against an
#: int64 overflow that no input reaches
_N_ATOMS = Dim("n_atoms", min=1, max=2**24)
_N_EDGES = Dim("n_edges", min=0, max=2**30)
_SIZES = {
    "positions": _N_ATOMS,
    "atomic_numbers": _N_ATOMS,
    "idx_i": _N_EDGES,
    "idx_j": _N_EDGES,
    "offsets": _N_EDGES,
    "local": _N_ATOMS,
}

#: tolerances of the check of a package against the eager potential, which differ
#: by round-off as AOTInductor fuses and reorders reductions
_TOLERANCES = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float64: dict(rtol=1e-9, atol=1e-10),
}


class LammpsPotential(nn.Module):
    """
    The energy of a ``NeuralNetworkPotential`` and its derivatives as functions of the
    flat tensors the LAMMPS pair style passes; :func:`deploy_lammps` compiles it.

    Inputs, for N atoms and E edges: ``positions [N, 3]``, ``atomic_numbers [N]``,
    ``idx_i [E]``, ``idx_j [E]``, ``offsets [E, 3]`` (Cartesian shift of atom j, zero
    for ghost atoms, which LAMMPS places at their image) and ``local [N]``, 1 for the
    atoms whose energy counts and 0 for ghost atoms that only complete their
    environments. Outputs (``LAMMPS_OUTPUTS``):

    - ``energy``: ``sum(local * e_i)``
    - ``energy_atom [N]``: ``local * e_i``
    - ``forces [N, 3]``: ``-d energy / d positions``, on ghost atoms too
    - ``virial [3, 3]``: ``sum_ij r_ij (x) d energy / d r_ij`` with a minus sign, i.e.
      LAMMPS' virial ``sum_i R_i (x) F_i``; the stress is ``-virial / volume``. It is
      taken from the edge vectors, so it needs no cell and holds with ghost atoms.

    Ghost atoms make the energy of a domain exact, as long as they cover the
    ``receptive_field`` around its local atoms; the energies, forces and virials of the
    domains add up to those of the whole system.

    The per-atom energy ``e_i`` is the per-atom output of the model's ``Atomwise``
    energy head plus the per-atom share of its ``AddOffsets`` postprocessors. Models
    whose energy has other terms (electrostatics, ZBL), that embed a total charge or
    spin, or whose representation is not SchNet, PaiNN or SO3net are rejected.

    The outputs carry no gradient; the potential is for inference only.
    """

    def __init__(self, model: NeuralNetworkPotential):
        """
        Args:
            model: a potential with an ``Atomwise`` energy head and ``Forces``; it is
                copied, not changed

        Raises:
            TypeError: if ``model`` is not a ``NeuralNetworkPotential``
            ValueError: if its energy cannot be deployed this way
        """
        super().__init__()
        if not isinstance(model, NeuralNetworkPotential):
            raise TypeError(
                f"LammpsPotential needs a NeuralNetworkPotential, not "
                f"{type(model).__name__}"
            )
        model = copy.deepcopy(model)

        forces = [m for m in model.output_modules if isinstance(m, Forces)]
        if len(forces) != 1:
            raise ValueError(
                "LammpsPotential needs a model with one Forces output module"
            )
        energy_key = forces[0].energy_key

        heads = []
        for m in model.output_modules:
            if isinstance(m, Forces):
                continue
            if isinstance(m, Atomwise) and m.output_key == energy_key:
                heads.append(m)
            elif getattr(m, "output_key", None) == energy_key:
                raise ValueError(
                    f"{type(m).__name__} adds to {energy_key!r}; only an Atomwise "
                    "energy can be split into the per-atom terms LAMMPS needs"
                )
            # outputs other than the energy are not deployed
        if len(heads) != 1:
            raise ValueError(
                f"LammpsPotential needs one Atomwise head for {energy_key!r}, "
                f"found {len(heads)}"
            )
        head = heads[0]
        if head.aggregation_mode != "sum" or head.n_out != 1:
            raise ValueError(
                "the Atomwise energy head must sum one output per atom, not "
                f"aggregate {head.n_out} with {head.aggregation_mode!r}"
            )

        for m in model.input_modules:
            # the virial comes from the edge vectors instead of a strain
            if not isinstance(m, (PairwiseDistances, Strain)):
                raise ValueError(
                    f"input module {type(m).__name__} is not supported for LAMMPS"
                )

        representation = model.representation
        if not isinstance(representation, (SchNet, PaiNN, SO3net)):
            raise ValueError(
                f"{type(representation).__name__} is not supported for LAMMPS; "
                "use SchNet, PaiNN or SO3net"
            )
        if len(representation.electronic_embeddings) > 0:
            raise ValueError(
                "a total charge or spin is a property of the whole system, which a "
                "LAMMPS domain does not see"
            )
        n_interactions = getattr(representation, "n_interactions", None) or len(
            representation.interactions
        )

        self.input_modules = nn.ModuleList(
            m for m in model.input_modules if isinstance(m, PairwiseDistances)
        )
        self.representation = representation
        self.head = head
        self.cutoff = float(representation.cutoff)
        #: how far the energy of an atom reaches: the ghost atoms of a domain must
        #: cover this distance around its local atoms
        self.receptive_field = n_interactions * self.cutoff

        atom_offset = _atom_offset(model, energy_key)
        self.add_offsets = atom_offset is not None
        self.register_buffer(
            "atom_offset", atom_offset if self.add_offsets else torch.zeros(1)
        )

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        offsets: torch.Tensor,
        local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            positions = positions.detach().requires_grad_()
            offsets = offsets.detach().requires_grad_()
            inputs = {
                properties.R: positions,
                properties.Z: atomic_numbers,
                properties.idx_i: idx_i,
                properties.idx_j: idx_j,
                properties.offsets: offsets,
            }
            for m in self.input_modules:
                inputs = m(inputs)
            inputs = self.representation(inputs)

            energy_atom = self.head.outnet(inputs["scalar_representation"]).squeeze(-1)
            if self.add_offsets:
                energy_atom = energy_atom + self.atom_offset[atomic_numbers]
            energy_atom = energy_atom * local
            energy = energy_atom.sum()
            dEdR, dEdr = torch.autograd.grad([energy], [positions, offsets])

        # d r_ij / d offsets_ij is the identity, so dEdr is the gradient w.r.t. r_ij
        virial = -torch.einsum("ea,eb->ab", inputs[properties.Rij].detach(), dEdr)
        return energy.detach(), energy_atom.detach(), -dEdR, virial


def _atom_offset(model: NeuralNetworkPotential, energy_key: str) -> torch.Tensor | None:
    """The per-element energy the model's postprocessors add to each atom, or None if
    they add nothing."""
    if not model.do_postprocessing:
        return None
    offset = None
    for pp in model.postprocessors:
        if isinstance(pp, (CastTo32, CastTo64)):
            # the dtype is chosen at deployment
            continue
        if isinstance(pp, AddOffsets) and pp._property == energy_key:
            if not pp.is_extensive:
                raise ValueError("an intensive energy has no per-atom terms")
            table = torch.zeros_like(pp.atomref)
            if pp.add_atomrefs:
                table = table + pp.atomref
            if pp.add_mean:
                table = table + pp.mean
            offset = table if offset is None else offset + table
        elif isinstance(pp, AddOffsets):
            # offsets of outputs other than the energy, which are not deployed
            continue
        else:
            raise ValueError(
                f"postprocessor {type(pp).__name__} is not supported for LAMMPS; "
                "supported are AddOffsets and the casts"
            )
    return offset.detach() if offset is not None else None


class _Deployed(nn.Module):
    """The traced graph of a :class:`LammpsPotential` with its parameters and buffers,
    called with positional tensors as ``torch.export`` and the C++ loader pass them."""

    def __init__(self, potential: LammpsPotential, traced: torch.fx.GraphModule):
        super().__init__()
        self.potential = potential
        # not `graph`, which would shadow the attribute of the exported GraphModule
        # that its constants are looked up under
        self.traced = traced

    def forward(self, positions, atomic_numbers, idx_i, idx_j, offsets, local):
        inputs = dict(
            zip(
                LAMMPS_INPUTS,
                (positions, atomic_numbers, idx_i, idx_j, offsets, local),
            )
        )
        return self.traced(
            dict(self.potential.named_parameters()),
            dict(self.potential.named_buffers()),
            inputs,
        )


def deploy_lammps(
    model: NeuralNetworkPotential,
    path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> str:
    """
    Compile ``model`` as a :class:`LammpsPotential` with AOTInductor and write the
    package that ``pair_style schnetpack`` loads to ``path``.

    The package is checked against the eager potential on a single atom without
    edges and on random systems with periodic shifts and ghost atoms before it is
    written; it holds its cutoff, receptive field, dtype, device and the torch and
    SchNetPack versions as metadata.

    Args:
        model: the potential, see :class:`LammpsPotential`
        path: where the package is written; ends in ``.pt2``
        device: the device LAMMPS runs the package on
        dtype: the dtype of positions and energies in the package

    Returns:
        ``path``
    """
    if not path.endswith(".pt2"):
        raise ValueError(f"a package path ends in .pt2, not {path!r}")
    device = torch.device(device)
    potential = LammpsPotential(model).to(device=device, dtype=dtype).eval()

    def forward(params, buffers, inputs):
        return functional_call(potential, (params, buffers), (), inputs)

    example = _example(potential, n_atoms=7, n_edges=20, seed=0)
    # recent torch guards on whether a dynamic size is 1 (see export_mace);
    # size-oblivious tracing assumes it is not, and the checks below hold the package
    # to a single atom and to no edges too
    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
        graph = trace(
            forward,
            dict(potential.named_parameters()),
            dict(potential.named_buffers()),
            example,
        )
        program = torch.export.export(
            _Deployed(potential, graph),
            tuple(example[k] for k in LAMMPS_INPUTS),
            dynamic_shapes=tuple({0: _SIZES[k]} for k in LAMMPS_INPUTS),
            strict=False,
        )

    metadata = {
        "cutoff": repr(potential.cutoff),
        "receptive_field": repr(potential.receptive_field),
        "dtype": str(dtype).removeprefix("torch."),
        "device": device.type,
        "torch_version": torch.__version__,
        "schnetpack_version": schnetpack.__version__,
        "inputs": ",".join(LAMMPS_INPUTS),
        "outputs": ",".join(LAMMPS_OUTPUTS),
    }
    # written into a temporary directory next to `path` first, so that a failed check
    # leaves no package; the package's file name names its contents
    directory = os.path.dirname(os.path.abspath(path))
    with tempfile.TemporaryDirectory(dir=directory, prefix=".spkdeploy-") as tmp:
        partial = os.path.join(tmp, os.path.basename(path))
        torch._inductor.aoti_compile_and_package(
            program,
            package_path=partial,
            inductor_configs={"aot_inductor.metadata": metadata},
        )
        _check(torch._inductor.aoti_load_package(partial), potential, dtype)
        os.replace(partial, path)
    return path


def _check(runner, potential: LammpsPotential, dtype: torch.dtype) -> None:
    """Compare a package to the eager potential on a single atom without edges and on
    random systems with periodic shifts and ghost atoms."""
    for n_atoms, n_edges, seed in ((1, 0, 1), (12, 40, 2), (40, 300, 3)):
        inputs = _example(potential, n_atoms, n_edges, seed)
        expected = potential(**inputs)
        actual = runner(*(inputs[k] for k in LAMMPS_INPUTS))
        for name, a, e in zip(LAMMPS_OUTPUTS, actual, expected):
            where = f"the package's {name} differs from the model on {n_atoms} atoms"
            where += f" and {n_edges} edges"
            torch.testing.assert_close(
                a, e, **_TOLERANCES[dtype], msg=lambda m, where=where: f"{where}: {m}"
            )


def _example(
    potential: LammpsPotential, n_atoms: int, n_edges: int, seed: int
) -> dict[str, torch.Tensor]:
    """Random inputs in the potential's dtype and device: atoms within a cutoff of
    each other, random edges with random shifts and some ghost atoms. Values to
    compare graphs on, not a physical system."""
    g = torch.Generator().manual_seed(seed)
    cutoff = potential.cutoff
    idx_i = torch.randint(n_atoms, (n_edges,), generator=g)
    # no edge from an atom to itself without a shift
    idx_j = (idx_i + torch.randint(1, max(n_atoms, 2), (n_edges,), generator=g)) % (
        n_atoms
    )
    local = (torch.rand(n_atoms, generator=g) < 0.7).double()
    local[0] = 1.0
    inputs = {
        "positions": torch.rand(n_atoms, 3, generator=g, dtype=torch.float64)
        * cutoff
        / 2,
        "atomic_numbers": torch.randint(1, 10, (n_atoms,), generator=g),
        "idx_i": idx_i,
        "idx_j": idx_j,
        "offsets": (torch.rand(n_edges, 3, generator=g, dtype=torch.float64) - 0.5)
        * cutoff
        / 2,
        "local": local,
    }
    dtype = potential.atom_offset.dtype
    device = potential.atom_offset.device
    return {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
    }

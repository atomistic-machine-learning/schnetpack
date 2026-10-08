"""The MACE teacher, checked against a scripted stand-in with MACE's interface
and, if one is given, against a real exported MACE-OFF archive."""

import os
import pickle

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from torch import nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.interfaces.ase_interface import atoms_to_batch
from schnetpack.train.teacher import MaceTeacher, teacher_key

from .conftest import molecules_batch, with_neighbors

R_MAX = 4.0
DIRECTION = (0.3, -0.2, 0.5)
#: per-element reference energies of H, C and O, as large as MACE-OFF's
E0 = (-13.6, -1029.8, -2041.8)


class AtomicEnergies(nn.Module):
    """MACE's ``AtomicEnergiesBlock``: the E0 table, one row per head on newer
    MACE versions."""

    def __init__(self, heads: int | None):
        super().__init__()
        table = torch.tensor(E0, dtype=torch.float64)
        if heads is not None:
            # only the first head is right; another one would show up as an offset
            table = torch.stack([table + 100.0 * head for head in range(heads)])
        self.register_buffer("atomic_energies", table)


class FakeMace(nn.Module):
    """MACE's interface on a toy potential.

    Like ``ScaleShiftMACE`` it reads a dict with a one-hot ``node_attrs``
    ordered by its ``atomic_numbers`` buffer, takes edge vectors as
    ``positions[edge_index[1]] - positions[edge_index[0]] + shifts``, adds the
    per-element reference energies of ``atomic_energies_fn`` (its first head)
    to the interaction energy in its own dtype, returns both, and keeps the
    graph of its forces only when called with ``training=True``. Its pair term
    depends on the direction of each edge, so a list handed in the wrong way
    round gives other energies.
    """

    def __init__(self, heads: int | None = None, interaction_energy: bool = True):
        super().__init__()
        self.with_interaction_energy = interaction_energy
        self.atomic_energies_fn = AtomicEnergies(heads)
        self.register_buffer("atomic_numbers", torch.tensor([1, 6, 8]))
        self.register_buffer("r_max", torch.tensor(R_MAX, dtype=torch.float64))
        self.register_buffer(
            "strength", torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        )
        self.register_buffer("direction", torch.tensor(DIRECTION, dtype=torch.float64))

    def forward(
        self, data: dict[str, torch.Tensor], training: bool = False
    ) -> dict[str, torch.Tensor | None]:
        positions = data["positions"]
        node_attrs = data["node_attrs"]
        edge_index = data["edge_index"]
        num_graphs = data["ptr"].numel() - 1
        vectors = positions[edge_index[1]] - positions[edge_index[0]] + data["shifts"]
        per_edge = pair_energy(
            vectors, (node_attrs @ self.strength)[edge_index[1]], self.r_max
        )
        per_edge = per_edge * (self.direction * vectors).sum(-1).add(1.0)
        node_interaction = torch.zeros_like(node_attrs[:, 0]).index_add(
            0, edge_index[0], per_edge
        )
        table = self.atomic_energies_fn.atomic_energies
        if table.dim() == 2:
            table = table[0]
        node_e0 = node_attrs @ table
        zeros = torch.zeros(num_graphs, dtype=positions.dtype, device=positions.device)
        e0 = zeros.index_add(0, data["batch"], node_e0)
        interaction = zeros.index_add(0, data["batch"], node_interaction)
        energy = e0 + interaction
        forces = grad(
            [energy.sum()], [positions], create_graph=training, retain_graph=training
        )[0]
        assert forces is not None
        output: dict[str, torch.Tensor | None] = {
            "energy": energy,
            "forces": -forces,
            "node_energy": node_e0 + node_interaction,
        }
        if self.with_interaction_energy:
            output["interaction_energy"] = interaction
        return output


def pair_energy(vectors, strength, r_max):
    lengths = torch.linalg.norm(vectors, dim=-1)
    return strength * (1 - (lengths / r_max) ** 2).clamp(min=0) ** 2


def reference(atoms_list, dtype=torch.float64):
    """FakeMace's energies, built from every periodic image of every ordered
    pair, without a neighbor list; forces and Hessians by autograd."""
    model = FakeMace()
    energies, positions_list = [], []
    for atoms in atoms_list:
        positions = torch.tensor(atoms.positions, dtype=dtype, requires_grad=True)
        columns = torch.tensor(
            [model.atomic_numbers.tolist().index(z) for z in atoms.numbers]
        )
        cell = torch.tensor(np.asarray(atoms.cell), dtype=dtype)
        reps = [range(-2, 3) if periodic else range(1) for periodic in atoms.pbc]
        energy = torch.tensor(E0, dtype=torch.float64)[columns].sum().to(dtype)
        for n in np.stack(np.meshgrid(*reps, indexing="ij"), -1).reshape(-1, 3):
            shift = torch.tensor(n, dtype=dtype) @ cell
            vectors = positions[None, :] - positions[:, None] + shift
            mask = vectors.norm(dim=-1) < R_MAX
            if not n.any():
                mask &= ~torch.eye(len(atoms), dtype=torch.bool)
            receiver = columns[None, :].expand(len(atoms), -1)[mask]
            per_edge = pair_energy(vectors[mask], model.strength[receiver], model.r_max)
            per_edge = per_edge * (vectors[mask] @ model.direction).add(1.0)
            energy = energy + per_edge.sum()
        energies.append(energy)
        positions_list.append(positions)
    return energies, positions_list


class RecordingMace(nn.Module):
    """Stands in for a loaded archive and records how it is called."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.atomic_numbers = model.atomic_numbers

    def forward(self, data, training=False):
        self.training_argument = training
        return self.model(data, training=training)


@pytest.fixture
def fake_path(tmp_path):
    path = str(tmp_path / "fake_mace.pt")
    torch.jit.save(torch.jit.script(FakeMace()), path)
    return path


def rattled(names):
    atoms = [molecule(name) for name in names]
    for seed, structure in enumerate(atoms):
        structure.rattle(0.05, seed=seed)
    return atoms


def periodic():
    """Hydrogen and oxygen in a cell smaller than the cutoff, so that most
    pairs are between periodic images."""
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.3, 0.1], [1.4, 1.6, 1.2]],
        cell=[2.8, 3.1, 2.9],
        pbc=True,
    )
    atoms.rattle(0.05, seed=3)
    return [atoms]


MOLECULES = ["CH4", "H2O", "C2H6"]


def test_matches_the_stand_in_on_molecules(fake_path):
    atoms = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))
    probe = torch.randn_like(batch[properties.R])

    targets = MaceTeacher(model_path=fake_path, cutoff=R_MAX, dtype="float64")(
        batch, probe
    )

    energies, positions = reference(atoms)
    forces = [-grad(e, r, create_graph=True)[0] for e, r in zip(energies, positions)]
    probes = torch.split(probe, [len(a) for a in atoms])
    hvp = [-grad(f, r, grad_outputs=v)[0] for f, r, v in zip(forces, positions, probes)]
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], torch.stack(energies).detach()
    )
    torch.testing.assert_close(
        targets[teacher_key(properties.forces)], torch.cat(forces).detach()
    )
    torch.testing.assert_close(targets[properties.teacher_hvp], torch.cat(hvp))


def test_handles_periodic_images(fake_path):
    atoms = periodic()
    batch = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))

    targets = MaceTeacher(model_path=fake_path, cutoff=R_MAX, dtype="float64")(batch)

    energies, positions = reference(atoms)
    forces = -grad(energies[0], positions[0])[0]
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], torch.stack(energies).detach()
    )
    torch.testing.assert_close(targets[teacher_key(properties.forces)], forces)


def test_runs_in_float32_by_default(fake_path):
    batch = molecules_batch()

    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)
    targets = teacher(batch, torch.randn_like(batch[properties.R]))

    assert teacher.dtype == torch.float32
    assert targets[teacher_key(properties.forces)].dtype == torch.float32
    assert targets[properties.teacher_hvp].dtype == torch.float32


def ethane_lattice(n=25, spacing=6.0):
    """``n`` ethane molecules in one structure, 8 n atoms: its absolute energy
    is tens of keV, far beyond float32's digits for a sub-meV target."""
    structure = Atoms()
    side = int(np.ceil(n ** (1 / 3)))
    for index in range(n):
        ethane = molecule("C2H6")
        ethane.rattle(0.05, seed=index)
        ethane.translate(spacing * np.array(np.unravel_index(index, (side,) * 3)))
        structure += ethane
    return [structure]


@pytest.mark.parametrize("heads", [None, 2])
def test_float32_energies_keep_their_digits(tmp_path, heads):
    """The E0s are summed in float64 and only the interaction energy carries
    float32 error: about 1e-4 eV here, against about 0.1 eV when the E0s are
    summed in float32 too. A per-head table contributes its first head."""
    path = str(tmp_path / "mace.pt")
    torch.jit.save(torch.jit.script(FakeMace(heads=heads)), path)
    atoms = ethane_lattice()
    batch64 = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))
    batch32 = with_neighbors(atoms_to_batch(atoms, dtype=torch.float32))

    energy = MaceTeacher(model_path=path, cutoff=R_MAX)(batch32)
    exact = MaceTeacher(model_path=path, cutoff=R_MAX, dtype="float64")(batch64)

    key = teacher_key(properties.energy)
    assert batch32[properties.n_atoms].item() == 200
    assert abs(energy[key] - exact[key]).item() < 1e-3


def test_an_archive_without_interaction_energy_falls_back_to_its_energy(tmp_path):
    path = str(tmp_path / "mace.pt")
    torch.jit.save(torch.jit.script(FakeMace(interaction_energy=False)), path)
    atoms = rattled(MOLECULES)
    batch = with_neighbors(atoms_to_batch(atoms, dtype=torch.float64))

    with pytest.warns(UserWarning, match="interaction_energy"):
        targets = MaceTeacher(model_path=path, cutoff=R_MAX, dtype="float64")(batch)

    energies, _ = reference(atoms)
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], torch.stack(energies).detach()
    )


@pytest.mark.parametrize("with_probe", [True, False])
def test_mace_keeps_the_graph_only_for_a_curvature_target(fake_path, with_probe):
    """MACE's forces keep their graph through its ``training`` argument."""

    batch = molecules_batch()
    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)
    teacher.model = RecordingMace(teacher.model)

    teacher(batch, torch.randn_like(batch[properties.R]) if with_probe else None)

    assert teacher.model.training_argument is with_probe


def test_a_cutoff_below_the_models_r_max_is_refused(fake_path):
    with pytest.raises(ValueError, match="r_max"):
        MaceTeacher(model_path=fake_path, cutoff=R_MAX - 0.5)


def test_elements_the_teacher_does_not_cover_are_refused(fake_path):
    batch = with_neighbors(atoms_to_batch(rattled(["CH4", "NH3"])))

    with pytest.raises(ValueError, match=r"\[7\]"):
        MaceTeacher(model_path=fake_path, cutoff=R_MAX)(batch)


def test_a_missing_archive_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        MaceTeacher(model_path=str(tmp_path / "missing.pt"))


def test_pickles_as_its_configuration(fake_path):
    batch = molecules_batch()
    teacher = MaceTeacher(model_path=fake_path, cutoff=R_MAX)

    restored = pickle.loads(pickle.dumps(teacher))

    torch.testing.assert_close(restored(batch), teacher(batch))


real_archive = pytest.mark.skipif(
    "SCHNETPACK_MACE_TEACHER" not in os.environ,
    reason="set SCHNETPACK_MACE_TEACHER to an exported MACE-OFF archive",
)


def reference_batches(dtype):
    """The structures the export script evaluated with MACE itself, next to the
    archive as ``<archive>.reference.npz``, with MACE's energies and forces."""
    reference = np.load(os.environ["SCHNETPACK_MACE_TEACHER"] + ".reference.npz")
    for name in sorted({key.split("/")[0] for key in reference.files}):
        atoms = Atoms(
            numbers=reference[f"{name}/numbers"],
            positions=reference[f"{name}/positions"],
            cell=reference[f"{name}/cell"],
            pbc=reference[f"{name}/pbc"],
        )
        batch = with_neighbors(atoms_to_batch([atoms], dtype=dtype))
        yield name, batch, reference[f"{name}/energy"], reference[f"{name}/forces"]


@real_archive
def test_a_real_mace_off_archive_reproduces_mace():
    teacher = MaceTeacher(
        model_path=os.environ["SCHNETPACK_MACE_TEACHER"], cutoff=5.0, dtype="float64"
    )
    for name, batch, energy, forces in reference_batches(torch.float64):
        targets = teacher(batch)

        torch.testing.assert_close(
            targets[teacher_key(properties.energy)],
            torch.tensor([float(energy)], dtype=torch.float64),
            rtol=0,
            atol=1e-6,
            msg=name,
        )
        torch.testing.assert_close(
            targets[teacher_key(properties.forces)],
            torch.from_numpy(forces),
            rtol=0,
            atol=1e-6,
            msg=name,
        )


@real_archive
def test_a_real_mace_off_archive_gives_curvature():
    """Curvature against a central difference of the forces, in float64."""
    teacher = MaceTeacher(
        model_path=os.environ["SCHNETPACK_MACE_TEACHER"], cutoff=5.0, dtype="float64"
    )
    _, batch, _, _ = next(reference_batches(torch.float64))
    probe = torch.randn_like(batch[properties.R])

    targets = teacher(batch, probe)

    eps = 1e-4
    forces_key = teacher_key(properties.forces)
    shifted = [
        teacher({**batch, properties.R: batch[properties.R] + sign * eps * probe})
        for sign in (1, -1)
    ]
    finite_difference = -(shifted[0][forces_key] - shifted[1][forces_key]) / (2 * eps)
    torch.testing.assert_close(
        targets[properties.teacher_hvp], finite_difference, rtol=1e-4, atol=1e-4
    )


@real_archive
def test_a_real_mace_off_archive_runs_in_float32():
    """The default: a float64 archive cast to float32, double backward included."""
    path = os.environ["SCHNETPACK_MACE_TEACHER"]
    _, batch64, energy, forces = next(reference_batches(torch.float64))
    batch = with_neighbors(
        {k: v.float() if v.is_floating_point() else v for k, v in batch64.items()}
    )
    probe = torch.randn_like(batch[properties.R])

    targets = MaceTeacher(model_path=path, cutoff=5.0)(batch, probe)
    reference = MaceTeacher(model_path=path, cutoff=5.0, dtype="float64")(
        batch64, probe.double()
    )

    torch.testing.assert_close(
        targets[teacher_key(properties.energy)],
        torch.tensor([float(energy)], dtype=torch.float64),
        rtol=0,
        atol=1e-4,
    )
    torch.testing.assert_close(
        targets[teacher_key(properties.forces)],
        torch.from_numpy(forces).float(),
        rtol=0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        targets[properties.teacher_hvp].double(),
        reference[properties.teacher_hvp],
        rtol=1e-3,
        atol=1e-2,
    )

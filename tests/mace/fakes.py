"""A scripted stand-in with MACE's interface, shared by the MACE tests."""

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from torch import nn
from torch.autograd import grad

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
        self.atomic_energies_fn = model.atomic_energies_fn
        self.r_max = model.r_max

    def forward(self, data, training=False):
        self.training_argument = training
        return self.model(data, training=training)


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

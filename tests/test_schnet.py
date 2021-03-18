import pytest
import torch

from schnetpack import Structure
import schnetpack as spk
import numpy as np
from ase.neighborlist import neighbor_list

from schnetpack.representation.schnet import SchNet

# TODO:make proper timing and golden tests


@pytest.fixture
def indexed_data(example_data, batch_size):
    Z = []
    R = []
    C = []
    seg_m = []
    seg_i = []
    ind_i = []
    ind_j = []
    ind_S = []
    Rij = []

    n_atoms = 0
    n_pairs = 0
    for i in range(len(example_data)):
        seg_m.append(n_atoms)
        atoms = example_data[i][0]
        atoms.set_pbc(False)
        Z.append(atoms.numbers)
        R.append(atoms.positions)
        C.append(atoms.cell)
        idx_i, idx_j, idx_S, rij = neighbor_list(
            "ijSD", atoms, 5.0, self_interaction=False
        )
        _, seg_im = np.unique(idx_i, return_counts=True)
        seg_im = np.cumsum(np.hstack((np.zeros((1,), dtype=np.int), seg_im)))
        seg_i.append(seg_im + n_pairs)
        ind_i.append(idx_i + n_atoms)
        ind_j.append(idx_j + n_atoms)
        ind_S.append(idx_S)
        Rij.append(rij.astype(np.float32))
        n_atoms += len(atoms)
        n_pairs += len(idx_i)
        if i + 1 >= batch_size:
            break
    seg_m.append(n_atoms)

    Z = np.hstack(Z)
    R = np.vstack(R).astype(np.float32)
    C = np.array(C).astype(np.float32)
    seg_m = np.hstack(seg_m)
    seg_i = np.hstack(seg_i)
    ind_i = np.hstack(ind_i)
    ind_j = np.hstack(ind_j)
    ind_S = np.vstack(ind_S)
    Rij = np.vstack(Rij)

    inputs = {
        Structure.Z: torch.tensor(Z),
        Structure.position: torch.tensor(R),
        Structure.cell: torch.tensor(C),
        "seg_m": torch.tensor(seg_m),
        "seg_i": torch.tensor(seg_i),
        "idx_j": torch.tensor(ind_j),
        "idx_i": torch.tensor(ind_i),
        "Rij": torch.tensor(Rij),
        Structure.cell_offset: torch.tensor(ind_S),
    }

    return inputs


# def test_cfconv(indexed_data, benchmark):
#     Z, R, seg_m, idx_i, idx_j, ind_S = (
#         indexed_data[Structure.Z],
#         indexed_data[Structure.R],
#         indexed_data["seg_m"],
#         indexed_data["idx_i"],
#         indexed_data["idx_j"],
#         indexed_data[Structure.cell_offset],
#     )
#
#     benchmark(cfconv, R, R[idx_j], idx_i, idx_j)


def test_schnet_new_coo(indexed_data, benchmark):
    Z, R, seg_m, seg_i, idx_i, idx_j, C, ind_S, r_ij = (
        indexed_data[Structure.Z],
        indexed_data[Structure.R],
        indexed_data["seg_m"],
        indexed_data["seg_i"],
        indexed_data["idx_i"],
        indexed_data["idx_j"],
        indexed_data[Structure.cell],
        indexed_data[Structure.cell_offset],
        indexed_data["Rij"],
    )

    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0)
    cutoff_fn = spk.nn.CosineCutoff(5.0)
    schnet = SchNet(
        n_atom_basis=128,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    benchmark(schnet, Z, r_ij, idx_i, idx_j)


def test_schnet_new_script(indexed_data, benchmark):
    Z, R, seg_m, seg_i, idx_i, idx_j, C, ind_S, r_ij = (
        indexed_data[Structure.Z],
        indexed_data[Structure.R],
        indexed_data["seg_m"],
        indexed_data["seg_i"],
        indexed_data["idx_i"],
        indexed_data["idx_j"],
        indexed_data[Structure.cell],
        indexed_data[Structure.cell_offset],
        indexed_data["Rij"],
    )

    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0)
    cutoff_fn = spk.nn.CosineCutoff(5.0)
    schnet = SchNet(
        n_atom_basis=128,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    schnet = torch.jit.script(schnet)
    schnet(Z, r_ij, idx_i, idx_j)

    benchmark(schnet, Z, r_ij, idx_i, idx_j)

import pytest
import torch

import schnetpack.properties as structure
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
    ind_i = np.hstack(ind_i)
    ind_j = np.hstack(ind_j)
    ind_S = np.vstack(ind_S)
    Rij = np.vstack(Rij)

    inputs = {
        structure.Z: torch.tensor(Z),
        structure.position: torch.tensor(R),
        structure.cell: torch.tensor(C),
        structure.idx_m: torch.tensor(seg_m),
        structure.idx_j: torch.tensor(ind_j),
        structure.idx_i: torch.tensor(ind_i),
        structure.Rij: torch.tensor(Rij),
    }

    return inputs


def test_schnet_new_coo(indexed_data, benchmark):
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0)
    cutoff_fn = spk.nn.CosineCutoff(5.0)
    schnet = SchNet(
        n_atom_basis=128,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    benchmark(schnet, indexed_data)


def test_schnet_new_script(indexed_data, benchmark):

    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0)
    cutoff_fn = spk.nn.CosineCutoff(5.0)
    schnet = SchNet(
        n_atom_basis=128,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )
    schnet = torch.jit.script(schnet)
    schnet(indexed_data)

    benchmark(schnet, indexed_data)

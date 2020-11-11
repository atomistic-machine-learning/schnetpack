import numpy as np
from ase.neighborlist import neighbor_list
import torch
import schnetpack.environment as env

from tests.fixtures import *


def test_single_atom(single_atom, simple_env, ase_env, torch_env):
    nbh_simple, offsets_simple = simple_env.get_environment(single_atom)
    nbh_ase, offsets_ase = ase_env.get_environment(single_atom)
    nbh_torch, offsets_torch = torch_env.get_environment(single_atom)

    assert nbh_simple.shape == nbh_ase.shape == nbh_torch.shape
    assert offsets_simple.shape == offsets_ase.shape == offsets_torch.shape
    assert np.allclose(nbh_simple, nbh_ase)
    assert np.allclose(nbh_simple, nbh_torch)
    assert np.allclose(offsets_simple, offsets_ase)
    assert np.allclose(offsets_simple, offsets_torch)


def test_two_atoms(two_atoms, simple_env, ase_env, torch_env):
    nbh_simple, offsets_simple = simple_env.get_environment(two_atoms)
    nbh_ase, offsets_ase = ase_env.get_environment(two_atoms)
    nbh_torch, offsets_torch = torch_env.get_environment(two_atoms)

    assert nbh_simple.shape == nbh_ase.shape == nbh_torch.shape
    assert offsets_simple.shape == offsets_ase.shape == offsets_torch.shape
    assert np.allclose(nbh_simple, nbh_ase)
    assert np.allclose(nbh_simple, nbh_torch)
    assert np.allclose(offsets_simple, offsets_ase)
    assert np.allclose(offsets_simple, offsets_torch)


def test_single_site_crystal_small_cutoff(crystal, simple_env, ase_env, torch_env):
    # assure that neighboring cells are not included
    cutoff = 0.5
    ase_env.cutoff = cutoff
    torch_env.cutoff = cutoff

    nbh_simple, offsets_simple = simple_env.get_environment(crystal)
    nbh_ase, offsets_ase = ase_env.get_environment(crystal)
    nbh_torch, offsets_torch = torch_env.get_environment(crystal)

    assert nbh_simple.shape == nbh_ase.shape == nbh_torch.shape
    assert offsets_simple.shape == offsets_ase.shape == offsets_torch.shape
    assert np.allclose(nbh_simple, nbh_ase)
    assert np.allclose(nbh_simple, nbh_torch)
    assert np.allclose(offsets_simple, offsets_ase)
    assert np.allclose(offsets_simple, offsets_torch)


def test_single_site_crystal_large_cutoff(crystal, ase_env, torch_env):
    cutoff = 2.0
    ase_env.cutoff = cutoff
    torch_env.cutoff = cutoff

    idx_i, idx_j, idx_S, dist = neighbor_list(
        "ijSd", crystal, ase_env.cutoff, self_interaction=False
    )

    nbh_ase, offsets_ase = ase_env.get_environment(crystal)
    nbh_torch, offsets_torch = torch_env.get_environment(crystal)

    # get number of neighbors from index vector
    n_nbh = (
        np.unique(
            np.hstack((idx_i, np.arange(crystal.get_global_number_of_atoms()))),
            return_counts=True,
        )[1]
        - 1
    )

    # get number of neighbors from nbh matrix
    n_nbh_ase_env = np.sum(nbh_ase >= 0, 1)
    n_nbh_torch_env = np.sum(nbh_torch >= 0, 1)

    # Compare the returned indices
    nbh_ref = idx_j.reshape(crystal.get_global_number_of_atoms(), -1)
    sorted_nbh_ref = np.sort(nbh_ref, axis=-1)
    sorted_nbh_ase = np.sort(nbh_ase, axis=-1)
    sorted_nbh_torch = np.sort(nbh_torch, axis=-1)

    assert n_nbh.shape == n_nbh_ase_env.shape == n_nbh_torch_env.shape
    assert np.allclose(n_nbh, n_nbh_ase_env)
    assert np.allclose(n_nbh, n_nbh_torch_env)
    assert np.allclose(sorted_nbh_ref, sorted_nbh_ase)
    assert np.allclose(sorted_nbh_ref, sorted_nbh_torch)
    assert np.allclose(sorted_nbh_ase, sorted_nbh_torch)


def test_collect_atom_triples(four_atoms, ase_env):
    # Get the environment
    nbh, offsets = ase_env.get_environment(four_atoms)

    # Generate general indices
    n_atoms, n_neighbors = nbh.shape

    idx_j = []
    idx_k = []
    for k in range(n_neighbors):
        for j in range(k + 1, n_neighbors):
            idx_j.append(j)
            idx_k.append(k)

    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)

    # Generate ase pair neighborhoods
    ase_nbh_j = nbh[:, idx_j]
    ase_nbh_k = nbh[:, idx_k]

    # Set up offset indices
    ase_off_idx_j = np.repeat(idx_j[None, :], n_atoms, axis=0)
    ase_off_idx_k = np.repeat(idx_k[None, :], n_atoms, axis=0)

    nbh_j, nbh_k, offset_idx_j, offset_idx_k = env.collect_atom_triples(nbh)

    assert np.allclose(ase_nbh_j, nbh_j)
    assert np.allclose(ase_nbh_k, nbh_k)
    assert np.allclose(ase_off_idx_j, offset_idx_j)
    assert np.allclose(ase_off_idx_k, offset_idx_k)


def test_collect_atom_triples_batch(four_atoms, ase_env):
    # Get the first environment (two atoms)
    ase_env.cutoff = 1.1
    nbh_1, offsets_1 = ase_env.get_environment(four_atoms)

    # Get the second environment (all_atoms)
    ase_env.cutoff = 3.0
    nbh_2, offsets_2 = ase_env.get_environment(four_atoms)

    # Pad to same size (assumes -1 padding)
    max_atoms = max(nbh_1.shape[0], nbh_2.shape[0])
    max_nbh = max(nbh_1.shape[1], nbh_2.shape[1])
    tmp_1 = -np.ones((max_atoms, max_nbh))
    tmp_2 = -np.ones((max_atoms, max_nbh))
    tmp_1[: nbh_1.shape[0], : nbh_1.shape[1]] = nbh_1
    tmp_2[: nbh_2.shape[0], : nbh_2.shape[1]] = nbh_2
    nbh_1 = tmp_1
    nbh_2 = tmp_2

    # Get masks and pair indices
    nbh_mask_1 = (nbh_1 >= 0).astype(np.int)
    nbh_mask_2 = (nbh_2 >= 0).astype(np.int)
    nbh_1_j, nbh_1_k, offset_idx_1_j, offset_idx_1_k = env.collect_atom_triples(nbh_1)
    nbh_2_j, nbh_2_k, offset_idx_2_j, offset_idx_2_k = env.collect_atom_triples(nbh_2)

    # Get pairwise masks
    mask_1_j = np.take_along_axis(nbh_mask_1, offset_idx_1_j, axis=1)
    mask_1_k = np.take_along_axis(nbh_mask_1, offset_idx_1_k, axis=1)
    mask_1_jk = mask_1_j * mask_1_k
    mask_2_j = np.take_along_axis(nbh_mask_2, offset_idx_2_j, axis=1)
    mask_2_k = np.take_along_axis(nbh_mask_2, offset_idx_2_k, axis=1)
    mask_2_jk = mask_2_j * mask_2_k

    # Generate batches and convert to torch
    batch_nbh = torch.LongTensor(np.array([nbh_1, nbh_2]))
    batch_nbh_mask = torch.LongTensor(np.array([nbh_mask_1, nbh_mask_2]))
    batch_nbh_j = torch.LongTensor(np.array([nbh_1_j, nbh_2_j]))
    batch_nbh_k = torch.LongTensor(np.array([nbh_1_k, nbh_2_k]))
    batch_offset_idx_j = torch.LongTensor(np.array([offset_idx_1_j, offset_idx_2_j]))
    batch_offset_idx_k = torch.LongTensor(np.array([offset_idx_1_k, offset_idx_2_k]))
    batch_mask_jk = torch.LongTensor(np.array([mask_1_jk, mask_2_jk]))

    # Collect triples via batch method
    (
        nbh_j,
        nbh_k,
        offset_idx_j,
        offset_idx_k,
        pair_mask,
    ) = env.collect_atom_triples_batch(batch_nbh, batch_nbh_mask)

    assert np.allclose(batch_nbh_j, nbh_j)
    assert np.allclose(batch_nbh_k, nbh_k)
    assert np.allclose(batch_offset_idx_j, offset_idx_j)
    assert np.allclose(batch_offset_idx_k, offset_idx_k)
    assert np.allclose(batch_mask_jk, pair_mask)

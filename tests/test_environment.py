import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list
import torch

import schnetpack.environment as env


@pytest.fixture
def single_atom():
    return Atoms([6], positions=[[0.0, 0.0, 0.0]])


@pytest.fixture
def two_atoms():
    return Atoms([6, 6], positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])


@pytest.fixture
def single_site_crystal():
    return Atoms([6], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)


@pytest.fixture
def two_site_crystal():
    return Atoms(
        [6, 6], positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], cell=np.eye(3), pbc=True
    )


@pytest.fixture(params=[0, 1])
def crystal(request, single_site_crystal, two_site_crystal):
    crystals = [single_site_crystal, two_site_crystal]
    yield crystals[request.param]


@pytest.fixture
def simple_env():
    return env.SimpleEnvironmentProvider()


@pytest.fixture
def ase_env():
    return env.AseEnvironmentProvider(10.0)


@pytest.fixture
def torch_env():
    # Select torch.device('cuda') to test on GPU
    return env.TorchEnvironmentProvider(10.0, device=torch.device("cpu"))


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
            np.hstack((idx_i, np.arange(crystal.get_number_of_atoms()))),
            return_counts=True,
        )[1]
        - 1
    )

    # get number of neighbors from nbh matrix
    n_nbh_ase_env = np.sum(nbh_ase >= 0, 1)
    n_nbh_torch_env = np.sum(nbh_torch >= 0, 1)

    # Compare the returned indices
    nbh_ref = idx_j.reshape(crystal.get_number_of_atoms(), -1)
    sorted_nbh_ref = np.sort(nbh_ref, axis=-1)
    sorted_nbh_ase = np.sort(nbh_ase, axis=-1)
    sorted_nbh_torch = np.sort(nbh_torch, axis=-1)

    assert n_nbh.shape == n_nbh_ase_env.shape == n_nbh_torch_env.shape
    assert np.allclose(n_nbh, n_nbh_ase_env)
    assert np.allclose(n_nbh, n_nbh_torch_env)
    assert np.allclose(sorted_nbh_ref, sorted_nbh_ase)
    assert np.allclose(sorted_nbh_ref, sorted_nbh_torch)
    assert np.allclose(sorted_nbh_ase, sorted_nbh_torch)

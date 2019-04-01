import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

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


def test_single_atom(single_atom, simple_env, ase_env):
    nbh_simple, offsets_simple = simple_env.get_environment(single_atom)
    nbh_ase, offsets_ase = ase_env.get_environment(single_atom)

    assert nbh_simple.shape == nbh_ase.shape
    assert offsets_simple.shape == offsets_ase.shape
    assert np.allclose(nbh_simple, nbh_ase)
    assert np.allclose(offsets_simple, offsets_ase)


def test_two_atoms(two_atoms, simple_env, ase_env):
    nbh_simple, offsets_simple = simple_env.get_environment(two_atoms)
    nbh_ase, offsets_ase = ase_env.get_environment(two_atoms)

    assert nbh_simple.shape == nbh_ase.shape
    assert offsets_simple.shape == offsets_ase.shape
    assert np.allclose(nbh_simple, nbh_ase)
    assert np.allclose(offsets_simple, offsets_ase)


def test_single_site_crystal_small_cutoff(crystal, simple_env, ase_env):
    # assure that neighboring cells are not included
    ase_env.cutoff = 0.5

    nbh_simple, offsets_simple = simple_env.get_environment(crystal)
    nbh_ase, offsets_ase = ase_env.get_environment(crystal)

    assert nbh_simple.shape == nbh_ase.shape
    assert offsets_simple.shape == offsets_ase.shape
    assert np.allclose(nbh_simple, nbh_ase)
    assert np.allclose(offsets_simple, offsets_ase)


def test_single_site_crystal_large_cutoff(crystal, ase_env):
    ase_env.cutoff = 0.7
    idx_i, idx_j, idx_S, dist = neighbor_list(
        "ijSd", crystal, ase_env.cutoff, self_interaction=False
    )

    nbh_ase, offsets_ase = ase_env.get_environment(crystal)

    # get number of neighbors from index vector
    n_nbh = (
        np.unique(
            np.hstack((idx_i, np.arange(crystal.get_number_of_atoms()))),
            return_counts=True,
        )[1]
        - 1
    )

    # get number of neighbors from nbh matrix
    n_nbh_env = np.sum(nbh_ase >= 0, 1)

    assert n_nbh.shape == n_nbh_env.shape
    assert np.allclose(n_nbh, n_nbh_env)

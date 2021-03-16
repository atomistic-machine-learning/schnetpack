import pytest
from ase import Atoms
import numpy as np


@pytest.fixture
def single_atom():
    return Atoms([6], positions=[[0.0, 0.0, 0.0]])


@pytest.fixture
def two_atoms():
    return Atoms([6, 6], positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])


@pytest.fixture
def four_atoms():
    return Atoms(
        [6, 6, 6, 6],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
    )


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
def h2o():
    return Atoms(positions=np.random.rand(3, 3), numbers=[1, 1, 8])


@pytest.fixture
def o2():
    return Atoms(positions=np.random.rand(2, 3), numbers=[8, 8])

import pytest
from ase import Atoms
import numpy as np
import torch
from schnetpack import Structure


@pytest.fixture
def single_atom():
    props = {
        Structure.Z: torch.tensor(np.array([6])),
        Structure.R: torch.tensor(np.array([[0.0, 0.0, 0.0]])),
        Structure.cell: torch.tensor(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ),
        Structure.pbc: torch.tensor(np.array([False, False, False])),
    }
    return props


@pytest.fixture
def two_atoms():
    props = {
        Structure.Z: torch.tensor(np.array([6, 2])),
        Structure.R: torch.tensor(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])),
        Structure.cell: torch.tensor(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ),
        Structure.pbc: torch.tensor(np.array([False, False, False])),
    }
    return props


@pytest.fixture
def four_atoms():
    props = {
        Structure.Z: torch.tensor(np.array([6, 2, 1, 7])),
        Structure.R: torch.tensor(
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
            )
        ),
        Structure.cell: torch.tensor(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ),
        Structure.pbc: torch.tensor(np.array([False, False, False])),
    }
    return props


@pytest.fixture
def single_site_crystal():
    props = {
        Structure.Z: torch.tensor(np.array([6])),
        Structure.R: torch.tensor(np.array([[0.0, 0.0, 0.0]])),
        Structure.cell: torch.tensor(np.eye(3)),
        Structure.pbc: torch.tensor(np.array([True, True, True])),
    }
    return props


@pytest.fixture
def two_site_crystal():
    props = {
        Structure.Z: torch.tensor(np.array([6, 1])),
        Structure.R: torch.tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])),
        Structure.cell: torch.tensor(np.eye(3)),
        Structure.pbc: torch.tensor(np.array([True, True, True])),
    }
    return props


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

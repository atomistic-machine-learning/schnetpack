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


@pytest.fixture
def environment_periodic():
    """
    System representing Argon gas in a box with periodic boundary conditions.
    Neighbor indices, shifts and distance vectors have been precomputed for a cutoff of 5 Angstrom
    """
    cutoff = 5.0
    props = {
        Structure.Z: torch.tensor(np.ones(5) * 18),
        Structure.R: torch.tensor(
            np.array(
                [
                    [1.8475400, 3.1888300, 2.88069500],
                    [5.1524600, 1.4996200, 4.04490500],
                    [3.9796300, 4.4159900, 5.77417500],
                    [4.5911200, 4.6424400, 1.67305500],
                    [2.1663800, 1.3575600, 6.32694500],
                ]
            )
        ),
        Structure.cell: torch.tensor(
            np.array([[7.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]])
        ),
        Structure.pbc: torch.tensor(np.array([True, True, True])),
    }
    neighbors = {
        Structure.idx_i: torch.LongTensor(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ]
        ),
        Structure.idx_j: torch.LongTensor(
            [
                4,
                1,
                3,
                3,
                2,
                1,
                4,
                0,
                4,
                3,
                2,
                0,
                2,
                3,
                4,
                1,
                4,
                4,
                3,
                3,
                1,
                0,
                2,
                0,
                1,
                2,
                1,
                0,
                4,
                1,
                2,
                1,
                0,
                3,
                0,
                2,
            ]
        ),
        Structure.Rij: torch.tensor(
            np.array(
                [
                    [0.31884, -1.83127, -4.55375],
                    [-3.69508, -1.68921, 1.16421],
                    [-4.25642, 1.45361, -1.20764],
                    [2.74358, 1.45361, -1.20764],
                    [2.13209, 1.22716, 2.89348],
                    [3.30492, -1.68921, 1.16421],
                    [0.31884, -1.83127, 3.44625],
                    [3.69508, 1.68921, -1.16421],
                    [-2.98608, -0.14206, 2.28204],
                    [-0.56134, 3.14282, -2.37185],
                    [-1.17283, 2.91637, 1.72927],
                    [-3.30492, 1.68921, -1.16421],
                    [-1.17283, -3.08363, 1.72927],
                    [-0.56134, -2.85718, -2.37185],
                    [4.01392, -0.14206, 2.28204],
                    [1.17283, 3.08363, -1.72927],
                    [-1.81325, -3.05843, 0.55277],
                    [-1.81325, 2.94157, 0.55277],
                    [0.61149, 0.22645, 3.89888],
                    [0.61149, 0.22645, -4.10112],
                    [1.17283, -2.91637, -1.72927],
                    [-2.13209, -1.22716, -2.89348],
                    [-0.61149, -0.22645, -3.89888],
                    [4.25642, -1.45361, 1.20764],
                    [0.56134, 2.85718, 2.37185],
                    [-0.61149, -0.22645, 4.10112],
                    [0.56134, -3.14282, 2.37185],
                    [-2.74358, -1.45361, 1.20764],
                    [-2.42474, 2.71512, -3.34611],
                    [-4.01392, 0.14206, -2.28204],
                    [1.81325, 3.05843, -0.55277],
                    [2.98608, 0.14206, -2.28204],
                    [-0.31884, 1.83127, -3.44625],
                    [2.42474, -2.71512, 3.34611],
                    [-0.31884, 1.83127, 4.55375],
                    [1.81325, -2.94157, -0.55277],
                ]
            )
        ),
        Structure.cell_offset: torch.tensor(
            np.array(
                [
                    [0, 0, -1],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, -1],
                    [-1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 1],
                    [0, -1, 0],
                ]
            )
        ),
    }
    return cutoff, props, neighbors


@pytest.fixture
def environment_nonperiodic():
    """
    System representing Argon gas in a box without periodic boundary conditions.
    Neighbor indices, shifts and distance vectors have been precomputed for a cutoff of 5 Angstrom
    """
    cutoff = 5.0
    props = {
        Structure.Z: torch.tensor(np.ones(5) * 18),
        Structure.R: torch.tensor(
            np.array(
                [
                    [1.8475400, 3.1888300, 2.88069500],
                    [5.1524600, 1.4996200, 4.04490500],
                    [3.9796300, 4.4159900, 5.77417500],
                    [4.5911200, 4.6424400, 1.67305500],
                    [2.1663800, 1.3575600, 6.32694500],
                ]
            )
        ),
        Structure.cell: torch.tensor(
            np.array([[7.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]])
        ),
        Structure.pbc: torch.tensor(np.array([False, False, False])),
    }
    neighbors = {
        Structure.idx_i: torch.LongTensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        ),
        Structure.idx_j: torch.LongTensor(
            [1, 2, 3, 4, 0, 2, 3, 4, 4, 3, 1, 0, 0, 1, 2, 0, 1, 2]
        ),
        Structure.Rij: torch.tensor(
            np.array(
                [
                    [3.30492, -1.68921, 1.16421],
                    [2.13209, 1.22716, 2.89348],
                    [2.74358, 1.45361, -1.20764],
                    [0.31884, -1.83127, 3.44625],
                    [-3.30492, 1.68921, -1.16421],
                    [-1.17283, 2.91637, 1.72927],
                    [-0.56134, 3.14282, -2.37185],
                    [-2.98608, -0.14206, 2.28204],
                    [-1.81325, -3.05843, 0.55277],
                    [0.61149, 0.22645, -4.10112],
                    [1.17283, -2.91637, -1.72927],
                    [-2.13209, -1.22716, -2.89348],
                    [-2.74358, -1.45361, 1.20764],
                    [0.56134, -3.14282, 2.37185],
                    [-0.61149, -0.22645, 4.10112],
                    [-0.31884, 1.83127, -3.44625],
                    [2.98608, 0.14206, -2.28204],
                    [1.81325, 3.05843, -0.55277],
                ]
            )
        ),
        Structure.cell_offset: torch.tensor(
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )
        ),
    }
    return cutoff, props, neighbors


@pytest.fixture(params=[0, 1])
def environment(request, environment_nonperiodic, environment_periodic):
    environments = [environment_nonperiodic, environment_periodic]
    return environments[request.param]

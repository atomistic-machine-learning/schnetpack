import pytest
import torch
import numpy as np

__all__ = [
    # input
    "schnet_batch",
    "max_atoms_in_batch",
    "neighbors",
    "neighbor_mask",
    "positions",
    "cell",
    "cell_offset",
    "r_ij",
    "f_ij",
    "random_atomic_env",
    "random_interatomic_distances",
    "random_input_dim",
    "random_output_dim",
    "random_shape",
    "random_float_input",
    "random_int_input",
    # output
    "schnet_output_shape",
    "interaction_output_shape",
    "cfconv_output_shape",
    "gaussian_smearing_shape",
]


# inputs
# from data
@pytest.fixture
def schnet_batch(example_loader):
    return next(iter(example_loader))


# components of batch
@pytest.fixture
def max_atoms_in_batch(schnet_batch):
    return schnet_batch["_positions"].shape[1]


@pytest.fixture
def neighbors(schnet_batch):
    return schnet_batch["_neighbors"]


@pytest.fixture
def neighbor_mask(schnet_batch):
    return schnet_batch["_neighbor_mask"]


@pytest.fixture
def positions(schnet_batch):
    return schnet_batch["_positions"]


@pytest.fixture
def cell(schnet_batch):
    return schnet_batch["_cell"]


@pytest.fixture
def cell_offset(schnet_batch):
    return schnet_batch["_cell_offset"]


@pytest.fixture
def r_ij(atom_distances, positions, neighbors, cell, cell_offset, neighbor_mask):
    return atom_distances(positions, neighbors, cell, cell_offset, neighbor_mask)


@pytest.fixture
def f_ij(gaussion_smearing_layer, r_ij):
    return gaussion_smearing_layer(r_ij)


@pytest.fixture
def random_atomic_env(batch_size, max_atoms_in_batch, n_filters):
    return torch.rand((batch_size, max_atoms_in_batch, n_filters))


@pytest.fixture
def random_interatomic_distances(batch_size, max_atoms_in_batch, cutoff):
    return (
        (1 - torch.rand((batch_size, max_atoms_in_batch, max_atoms_in_batch - 1)))
        * 2
        * cutoff
    )


@pytest.fixture
def random_input_dim(random_shape):
    return random_shape[-1]


@pytest.fixture
def random_output_dim():
    return np.random.randint(1, 20, 1).item()


@pytest.fixture
def random_shape():
    return list(np.random.randint(1, 8, 3))


@pytest.fixture
def random_float_input(random_shape):
    return torch.rand(random_shape, dtype=torch.float32)


@pytest.fixture
def random_int_input(random_shape):
    return torch.randint(0, 20, random_shape)


# outputs
# spk.representation
@pytest.fixture
def schnet_output_shape(batch_size, max_atoms_in_batch, n_atom_basis):
    return [batch_size, max_atoms_in_batch, n_atom_basis]


@pytest.fixture
def interaction_output_shape(batch_size, max_atoms_in_batch, n_filters):
    return [batch_size, max_atoms_in_batch, n_filters]


@pytest.fixture
def cfconv_output_shape(batch_size, max_atoms_in_batch, n_atom_basis):
    return [batch_size, max_atoms_in_batch, n_atom_basis]


# spk.nn
@pytest.fixture
def gaussian_smearing_shape(batch_size, max_atoms_in_batch, n_gaussians):
    return [batch_size, max_atoms_in_batch, max_atoms_in_batch - 1, n_gaussians]

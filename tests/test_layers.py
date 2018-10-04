import os
import numpy as np
import pytest
import torch
from torch.optim import Adam
from torch.nn.modules import MSELoss

from schnetpack.nn.cfconv import CFConv
from schnetpack.representation.schnet import SchNet, SchNetInteraction
from schnetpack.data import Structure


@pytest.fixture
def batchsize():
    return 4

@pytest.fixture
def n_atom_basis():
    return 128

@pytest.fixture
def n_atoms():
    return 19

@pytest.fixture
def n_spatial_basis():
    return 25

@pytest.fixture
def n_filters():
    return 128

@pytest.fixture
def atomic_env(batchsize, n_atoms, n_filters):
    return torch.rand((batchsize, n_atoms, n_filters))

@pytest.fixture
def atomic_numbers(batchsize, n_atoms):
    atoms = np.random.randint(1, 9, (1, n_atoms))
    return torch.LongTensor(np.repeat(atoms, batchsize, axis=0))

@pytest.fixture
def positions(batchsize, n_atoms):
    return torch.rand((batchsize, n_atoms, 3))

@pytest.fixture
def cell(batchsize):
    return torch.zeros((batchsize, 3, 3))

@pytest.fixture
def cell_offset(batchsize, n_atoms):
    return torch.zeros((batchsize, n_atoms, n_atoms - 1, 3))

@pytest.fixture
def neighbors(batchsize, n_atoms):
    neighbors = np.array([range(n_atoms)]*n_atoms)
    neighbors = neighbors[~np.eye(neighbors.shape[0], dtype=bool)].reshape(
        neighbors.shape[0], -1)[np.newaxis, :]
    return torch.LongTensor(np.repeat(neighbors, batchsize, axis=0))

@pytest.fixture
def neighbor_mask(batchsize, n_atoms):
    return torch.ones((batchsize, n_atoms, n_atoms - 1))

@pytest.fixture
def schnet_batch(atomic_numbers, positions, cell, cell_offset, neighbors, neighbor_mask):
    inputs = {}
    inputs[Structure.Z] = atomic_numbers
    inputs[Structure.R] = positions
    inputs[Structure.cell] = cell
    inputs[Structure.cell_offset] = cell_offset
    inputs[Structure.neighbors] = neighbors
    inputs[Structure.neighbor_mask] = neighbor_mask
    return inputs

@pytest.fixture
def distances(batchsize, n_atoms):
    return torch.rand((batchsize, n_atoms, n_atoms - 1))


def assert_params_changed(model, input, exclude=[]):
    """
    Check if all model-parameters are updated when training.

    Args:
        model (torch.nn.Module): model to test
        data (torch.utils.data.Dataset): input dataset
        exclude (list): layers that are not necessarily updated
    """
    # save state-dict
    torch.save(model.state_dict(), 'before')
    # do one training step
    optimizer = Adam(model.parameters())
    loss_fn = MSELoss()
    pred = model(*input)
    loss = loss_fn(pred, torch.rand(pred.shape))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # check if all trainable parameters have changed
    after = model.state_dict()
    before = torch.load('before')
    for key in before.keys():
        if sum([key.startswith(exclude_layer) for exclude_layer in exclude]) != 0:
            continue
        assert (before[key] != after[key]).any(), 'Not all Parameters have been updated!'

def assert_equal_shape(model, batch, out_shape):
    """
    Check if the model returns the desired output shape.

    Args:
        model (nn.Module): model that needs to be tested
        test_batch (list): input data
        out_shape (list): desired output shape
    """
    pred = model(*batch)
    assert list(pred.shape) == out_shape, 'Model does not return expected shape!'

def test_parameter_update_schnet(schnet_batch):
    model = SchNet()
    schnet_batch = [schnet_batch]
    assert_params_changed(model, schnet_batch, exclude=['distance_expansion'])

def test_shape_schnet(schnet_batch, batchsize, n_atoms, n_atom_basis):
    schnet_batch = [schnet_batch]
    model = SchNet(n_atom_basis=n_atom_basis)

    assert_equal_shape(model, schnet_batch, [batchsize, n_atoms, n_atom_basis])

def test_shape_schnetinteraction(batchsize, n_atoms, n_atom_basis, n_spatial_basis, n_filters,
                                 atomic_env, distances, neighbors, neighbor_mask):
    model = SchNetInteraction(n_atom_basis, 1, n_filters)
    out_shape = [batchsize, n_atoms, n_filters]
    inputs = [atomic_env, distances, neighbors, neighbor_mask]
    assert_equal_shape(model, inputs, out_shape)

def teardown_module():
    """
    Remove artifacts that have been created during testing.
    """
    if os.path.exists('before'):
        os.remove('before')

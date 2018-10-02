import os
import numpy as np
import pytest
import torch
from torch.optim import Adam
from torch.nn.modules import MSELoss

from schnetpack.nn.cfconv import CFConv
from schnetpack.representation.schnet import SchNet
from schnetpack.data import Structure


def assert_params_changed(model, test_batch, exclude=[]):
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
    pred = model(test_batch)
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

def assert_equal_shape(model, test_batch, out_shape):
    """
    Check if the model returns the desired output shape.

    Args:
        model (nn.Module): model that needs to be tested
        test_batch (dict): input data
        out_shape (list): desired output shape
    """
    pred = model(test_batch)
    assert list(pred.shape) == out_shape, 'Model does not return expected shape!'


@pytest.fixture
def test_batch():
    inputs = {}
    inputs[Structure.Z] = torch.LongTensor([[1]*10 + [6]*7 + [8]*2]*4)
    inputs[Structure.R] = torch.rand((4, 19, 3))
    inputs[Structure.cell] = torch.zeros((4, 3, 3))
    inputs[Structure.cell_offset] = torch.zeros((4, 19, 18, 3))
    neighbors = np.array([range(19)]*19)
    neighbors = neighbors[~np.eye(neighbors.shape[0], dtype=bool)].reshape(
        neighbors.shape[0], -1)[np.newaxis, :]
    neighbors = np.vstack((neighbors, neighbors, neighbors, neighbors))
    inputs[Structure.neighbors] = torch.LongTensor(neighbors)
    inputs[Structure.neighbor_mask] = torch.ones((4, 19, 18))

    return inputs

def test_parameter_update_schnet(test_batch):
    model = SchNet()
    assert_params_changed(model, test_batch, exclude=['distance_expansion'])

def test_shape_schnet(test_batch):
    batch_size = 4
    n_atom_basis = 100
    n_atoms = 19

    model = SchNet(n_atom_basis=n_atom_basis)

    assert_equal_shape(model, test_batch, [batch_size, n_atoms, n_atom_basis])

def teardown_module():
    """
    Remove artifacts that have been created during testing.
    """
    if os.path.exists('before'):
        os.remove('before')

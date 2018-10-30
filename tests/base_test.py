import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.nn.modules import MSELoss


def assert_params_changed(model, input, exclude=[], return_label='y'):
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
    if type(pred) == dict:
        pred = pred[return_label]
    loss = loss_fn(pred, torch.rand(pred.shape))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # check if all trainable parameters have changed
    after = model.state_dict()
    before = torch.load('before')
    for key in before.keys():
        if np.array([key.startswith(exclude_layer) for exclude_layer in exclude]).any():
            continue
        assert (before[key] != after[key]).any(), '{} layer has not been updated!'.format(key)


def assert_equal_shape(model, batch, out_shape):
    """
    Check if the model returns the desired output shape.

    Args:
        model (nn.Module): model that needs to be tested
        batch (list): input data
        out_shape (list): desired output shape
    """
    pred = model(*batch)
    assert list(pred.shape) == out_shape, 'Model does not return expected shape!'
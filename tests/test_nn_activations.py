import torch
import numpy as np

import schnetpack as spk
from .assertions import (
    assert_shape_invariance,
    assert_nn_equal_params,
    assert_params_changed,
)


from .fixtures import *


# ssp activation
def test_activation_softplus():
    # simple tensor
    x = torch.tensor([0.0, 1.0, 0.5, 2.0])
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, spk.nn.shifted_softplus(x), atol=0.0, rtol=1.0e-7)
    # random tensor
    torch.manual_seed(42)
    x = torch.randn((10, 5), dtype=torch.double)
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, spk.nn.shifted_softplus(x), atol=0.0, rtol=1.0e-7)
    x = 10 * torch.randn((10, 5), dtype=torch.double)
    expt = torch.log(1.0 + torch.exp(x)) - np.log(2)
    assert torch.allclose(expt, spk.nn.shifted_softplus(x), atol=0.0, rtol=1.0e-7)


def test_shape_ssp():
    assert_shape_invariance(spk.nn.shifted_softplus)


# swish activation
def test_shape_swish(swish, random_input_dim):
    assert_shape_invariance(swish, in_data=torch.rand(random_input_dim))


def test_swish_parameter_update(swish, random_float_input):
    assert_params_changed(swish, random_float_input)


# utility functions
def test_activation_getter(swish, random_input_dim):
    make_a_swish = spk.nn.activation_factory(spk.nn.Swish)
    swish2 = make_a_swish(**dict(n_features=random_input_dim))

    assert_nn_equal_params(swish, swish2)

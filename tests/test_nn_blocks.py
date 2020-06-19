import pytest
import schnetpack as spk
from .assertions import assert_params_changed, assert_shape_invariance

from .fixtures import *


# nn.ResidualBlock
def test_residual_shape_invariant(residual_block, random_float_input):
    assert_shape_invariance(residual_block, random_float_input)


# nn.ResidualStack
def test_residual_stack_shape_invariant(residual_stack, random_float_input):
    assert_shape_invariance(residual_stack, random_float_input)

"""Switching a model to train mode for a block, since curvature needs forces with
a graph. The mode is restored afterwards, also when the block fails."""

import pytest
from torch import nn

from schnetpack import properties
from schnetpack.model import train_mode
from schnetpack.model.utils import forces_need_train_mode


def test_train_mode_is_switched_for_the_block_and_restored():
    model = nn.Linear(1, 1).eval()

    with train_mode(model):
        inside = model.training

    assert inside is True and model.training is False


def test_train_mode_is_restored_after_an_error():
    model = nn.Linear(1, 1).eval()

    with pytest.raises(RuntimeError), train_mode(model):
        raise RuntimeError("forward failed")

    assert model.training is False


def test_only_forces_taken_as_a_position_derivative_need_train_mode():
    """SchNetPack's Forces keep their graph only in train mode; forces predicted
    directly keep it in eval mode too. A model that says nothing is assumed to
    need it."""
    conservative, direct, unknown = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1)
    conservative.required_derivatives = [properties.R]
    direct.required_derivatives = []

    assert forces_need_train_mode(conservative)
    assert not forces_need_train_mode(direct)
    assert forces_need_train_mode(unknown)

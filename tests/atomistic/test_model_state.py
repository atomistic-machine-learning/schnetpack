"""Switching a model to train mode for a block, since curvature needs forces with
a graph. The mode is restored afterwards, also when the block fails."""

import pytest
from torch import nn

from schnetpack.model import train_mode


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

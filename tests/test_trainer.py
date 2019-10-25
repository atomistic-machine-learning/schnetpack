import os
import pytest
from .fixtures.data import *
from .fixtures.train import *
from .fixtures.model import *
import schnetpack as spk
import torch
from numpy.testing import assert_array_almost_equal


class TestTrainer:
    def test_train(
        self, trainer, test_loader, modeldir, keep_n_checkpoints, checkpoint_interval
    ):
        # forward pass before training
        for batch in test_loader:
            init_result = trainer._model(batch)
            break

        # train
        trainer.train("cpu", n_epochs=1)
        assert trainer.epoch == 1

        # restore and train
        trainer.restore_checkpoint()
        trainer.train("cpu", n_epochs=1)
        assert trainer.epoch == 2

        # test if output changes
        for batch in test_loader:
            after_train_result = trainer._model(batch)
            break
        for key in init_result.keys():
            assert (init_result[key] - after_train_result[key]).abs().sum() >= 1e-5

        # test checkpoints
        trainer.train("cpu", n_epochs=3)
        assert trainer.epoch == 5
        trainer.restore_checkpoint(4)
        assert trainer.epoch == 4

        with pytest.raises(Exception):
            trainer.restore_checkpoint(1)

        trainer.restore_checkpoint(5)
        assert trainer.epoch == 5


def test_loss_fn():
    rho = dict(stress=0.1, derivative=1, property=0.9, contributions=0.2)
    property_names = dict(
        stress="stress_tensor",
        derivative="forces",
        property="energy",
        contributions="atomic_energy",
    )
    loss_fn = spk.utils.tradeoff_loss_fn(rho, property_names)
    target = {
        property_names["property"]: torch.rand(10, 1),
        property_names["derivative"]: torch.rand(10, 8, 3),
        property_names["contributions"]: torch.rand(10, 8, 1),
        property_names["stress"]: torch.rand(10, 3, 3),
    }
    pred = {key: torch.zeros_like(val) for key, val in target.items()}

    loss = loss_fn(target, pred)

    val_loss = 0.0
    for key, val in rho.items():
        val_loss += torch.mean(target[property_names[key]] ** 2) * val

    assert_array_almost_equal(loss, val_loss)

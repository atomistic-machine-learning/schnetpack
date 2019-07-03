import os
import pytest
from .fixtures.data import *
from .fixtures.train import *
from .fixtures.model import *


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

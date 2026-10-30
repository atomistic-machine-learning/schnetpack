import os

import hydra
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir

import schnetpack

CONFIG_DIR = os.path.join(os.path.dirname(schnetpack.__file__), "configs")


@pytest.mark.parametrize("trainer", ["default_trainer", "debug_trainer"])
def test_trainer_config_builds_a_trainer(trainer):
    with initialize_config_dir(CONFIG_DIR, version_base="1.2"):
        config = compose("train", overrides=[f"trainer={trainer}"])

    assert isinstance(hydra.utils.instantiate(config.trainer, logger=False), pl.Trainer)

import hydra
from typing import Dict
from omegaconf import DictConfig, OmegaConf

import uuid

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))

from dataclasses import dataclass

# Logger configs
@dataclass
class TensorboardLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.tensorboard.TensorBoardLogger"
    save_dir: str = "tensorboard/"
    name: str = "default"


@dataclass
class CSVLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.csv_logs.CSVLogger"
    save_dir: str = "."
    name: str = "csv/"


@dataclass
class AimLoggerConfig:
    _target_: str = "aim.pytorch_lightning.AimLogger"
    repo: str = "${hydra:runtime.cwd}/${run_dir}"
    experiment: str = "${name}"


# Trainer
from hydra_configs.pytorch_lightning.trainer import TrainerConf


@dataclass
class DefaultTrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"

    gpus: int = 1
    precision: int = 32

    min_epochs: int = 1
    max_epochs: int = 100000

    progress_bar_refresh_rate: int = 10
    terminate_on_nan: bool = True


def register_configs():
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    # register logger configs
    cs.store(group="logger", name="csv", node=CSVLoggerConfig)
    cs.store(group="logger", name="tensorboard", node=TensorboardLoggerConfig)
    cs.store(group="logger", name="aim", node=AimLoggerConfig)

    # register trainer configs
    cs.store(group="trainer", name="default", node=DefaultTrainerConf)

from typing import List, Union, Dict

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import yaml
from aim.pytorch_lightning import AimLogger


def empty(*args, **kwargs):
    pass


def todict(config: Union[DictConfig, Dict]):
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    print(config_dict)
    return config_dict


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
) -> None:
    """
    This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["run_dir"] = config["run_dir"]
    hparams["trainer"] = todict(config["trainer"])
    hparams["model"] = todict(config["model"])
    hparams["data"] = todict(config["data"])
    if "callbacks" in config:
        hparams["callbacks"] = todict(config["callbacks"])

    # save sizes of each dataset
    datamodule.setup()
    if hasattr(datamodule, "data_train") and datamodule.data_train:
        hparams["data"]["train_size"] = len(datamodule.data_train)
    if hasattr(datamodule, "data_val") and datamodule.data_val:
        hparams["data"]["val_size"] = len(datamodule.data_val)
    if hasattr(datamodule, "data_test") and datamodule.data_test:
        hparams["data"]["test_size"] = len(datamodule.data_test)

    # save number of model parameters
    hparams["model"]["params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model"]["params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model"]["params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty

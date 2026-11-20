import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

__all__ = ["log_hyperparameters"]


@rank_zero_only
def log_hyperparameters(config: DictConfig, trainer: pl.Trainer) -> None:
    """Send the Hydra config to every logger of the trainer."""
    for logger in trainer.loggers:
        logger.log_hyperparams(config)

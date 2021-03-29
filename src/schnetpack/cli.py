import hydra
import logging
from omegaconf import DictConfig

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything


log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train")
def train(config: DictConfig):
    if config.get("print_config"):
        print(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model, datamodule=datamodule
    )
    print(model.hparams)

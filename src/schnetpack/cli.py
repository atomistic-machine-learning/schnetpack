import logging
import os
import uuid
from typing import List
from pathlib import Path
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from schnetpack.utils.script import log_hyperparameters, print_config
from schnetpack.utils import str2class

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))


@hydra.main(config_path="configs", config_name="train")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(
        """
   _____      __    _   __     __  ____             __  
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<   
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|                                                          
    """
    )
    if config.get("print_config"):
        print_config(config, resolve=True)

    if not ("model" in config and "data" in config):
        log.error(
            f"""
        Config incomplete! You have to specify at least `data` and `model`! 
        For an example, try one of our pre-defined experiments:
        > spktrain data_dir=/data/will/be/here +experiment=qm9
        """
        )
        return

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    scheduler_cls = (
        str2class(config.model.scheduler_cls) if config.model.scheduler_cls else None
    )
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        datamodule=datamodule,
        optimizer_cls=str2class(config.model.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)

                # set run_id for AimLogger
                if lg_conf["_target_"] == "aim.pytorch_lightning.AimLogger":
                    from aim import Session

                    sess = Session(
                        repo=l._repo_path,
                        experiment=l._experiment_name,
                        flush_frequency=l._flush_frequency,
                        system_tracking_interval=l._system_tracking_interval,
                        run=config.run_id,
                    )
                    l._aim_session = sess

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.name, config.run_id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=model, trainer=trainer)

    # Train the model
    log.info("Starting training.")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test()

    # Remove temporary files if existing
    env_dir = Path(os.path.join(os.getcwd(), "environments"))
    if env_dir.exists() and env_dir.is_dir():
        shutil.rmtree(env_dir)

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

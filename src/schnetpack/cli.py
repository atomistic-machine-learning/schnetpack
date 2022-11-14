import logging
import os
import uuid
import tempfile
import socket
from typing import List

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import schnetpack as spk
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config
from schnetpack.data import BaseAtomsData, AtomsLoader
from schnetpack.train import PredictionWriter

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

header = """
   _____      __    _   __     __  ____             __  
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<   
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|                                                          
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            f"Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            f"""
        Config incomplete! You have to specify at least `data` and `model`! 
        For an example, try one of our pre-defined experiments:
        > spktrain data_dir=/data/will/be/here +experiment=qm9
        """
        )
        return

    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.trainer.resume_from_checkpoint is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.trainer.resume_from_checkpoint = "checkpoints/last.ckpt"

        if config.trainer.resume_from_checkpoint is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.trainer.resume_from_checkpoint)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    if config.get("print_config"):
        print_config(config, resolve=False)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info(f"Seed randomly...")
        seed_everything(workers=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
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

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Train the model
    log.info("Starting training.")
    trainer.fit(model=task, datamodule=datamodule)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info(f"Store best model")
    best_task = type(task).load_from_checkpoint(best_path)
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")


@hydra.main(config_path="configs", config_name="predict", version_base="1.2")
def predict(config: DictConfig):
    log.info(f"Load data from `{config.data.datapath}`")
    dataset: BaseAtomsData = hydra.utils.instantiate(config.data)
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)

    model = torch.load("best_model")

    class WrapperLM(LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return model(x)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.outputdir, write_interval=config.write_interval
            )
        ],
        default_root_dir=".",
        resume_from_checkpoint="checkpoints/last.ckpt",
        _convert_="partial",
    )
    trainer.predict(WrapperLM(model), dataloaders=loader)

"""spktrain: train a model from the Hydra configs (``configs/train.yaml``), then
test its best checkpoint and store the best model."""

import json
import logging
import os
import random
import socket

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger

from schnetpack.lightning import AtomisticTask
from schnetpack.lightning.utils import (
    load_task_from_checkpoint,
    log_hyperparameters,
    trainer_fit_kwargs_for_checkpoint,
)
from schnetpack.utils import str2class
from schnetpack.utils.script import print_config

log = logging.getLogger(__name__)

header = r"""
   _____      __    _   __     __  ____             __
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|
"""


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            "Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            """
        Config incomplete! You have to specify at least `data` and `model`!
        For an example, try one of our pre-defined experiments:
        > spktrain experiment=qm9_atomwise
        """
        )
        return

    # seed of torch, numpy and python's random; drawn before config.yaml is saved
    if config.get("seed") is None:
        config.seed = random.randint(0, 2**32 - 1)
        log.info(f"Seed randomly with <{config.seed}>")
    else:
        log.info(f"Seed with <{config.seed}>")

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
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"

        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    seed_everything(seed=config.seed, workers=True)

    if config.get("print_config"):
        print_config(config, resolve=False)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    datamodule = _datamodule(config)

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: list[Logger] = []

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
    fit_kwargs = (
        trainer_fit_kwargs_for_checkpoint(trainer)
        if config.run.ckpt_path is not None
        else {}
    )
    trainer.fit(
        model=task,
        datamodule=datamodule,
        ckpt_path=config.run.ckpt_path,
        **fit_kwargs,
    )

    # Lightning's EarlyStopping stops training silently on non-finite metrics
    # (check_finite=True), so fail explicitly instead of testing the model
    for early_stopping in trainer.early_stopping_callbacks:
        if not early_stopping.check_finite:
            continue
        value = trainer.callback_metrics.get(early_stopping.monitor)
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(
                f"Training stopped at epoch {early_stopping.stopped_epoch} because the "
                f"monitored metric became non-finite "
                f"({early_stopping.monitor} = {value.item()}). "
                "Skipping testing and model export."
            )

    # Load the best checkpoint through the compatibility helper (it handles
    # `weights_only` across PL versions) and test that task directly, instead
    # of having Lightning re-load the checkpoint internally via
    # ckpt_path="best", whose weights_only handling is version-dependent.
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")
    log.info("Starting testing.")
    best_task = test_checkpoint(trainer, datamodule, type(task), best_path)

    # Store best model
    log.info("Store best model")
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")


def _datamodule(config: DictConfig) -> LightningDataModule:
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    return hydra.utils.instantiate(
        config.data,
        train_sampler_cls=(
            str2class(config.data.train_sampler_cls)
            if config.data.train_sampler_cls
            else None
        ),
    )


def test_checkpoint(
    trainer: Trainer,
    datamodule: LightningDataModule,
    task_cls: type,
    ckpt_path: str,
    path: str = "test_metrics.json",
) -> AtomisticTask:
    """Test the task of a checkpoint on the test split and write the metrics, each
    with the unit of its target property (none for the loss), to ``path``."""
    task = load_task_from_checkpoint(task_cls, ckpt_path)
    (metrics,) = trainer.test(model=task, datamodule=datamodule)
    units = {
        f"test_{output.target_property}_{name}": datamodule.dataset.units.get(
            output.target_property
        )
        for output in task.outputs
        for name in output.metrics["test"]
    }
    result = {
        "checkpoint": os.path.relpath(ckpt_path),
        "epoch": torch.load(ckpt_path, map_location="cpu", weights_only=False)["epoch"],
        "n_test": len(datamodule.test_idx),
        "metrics": {k: {"value": v, "unit": units.get(k)} for k, v in metrics.items()},
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Test metrics stored at {os.path.abspath(path)}")
    return task

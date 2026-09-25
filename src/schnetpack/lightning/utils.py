import inspect
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

__all__ = [
    "log_hyperparameters",
    "load_task_from_checkpoint",
    "trainer_fit_kwargs_for_checkpoint",
]


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    This saves Hydra config using Lightning loggers.
    """

    # send hparams to all loggers
    trainer.logger.log_hyperparams(config)

    # disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty


def trainer_fit_kwargs_for_checkpoint(trainer) -> dict:
    """
    Return kwargs for ``Trainer.fit`` when resuming from a checkpoint.

    PyTorch >= 2.6 defaults ``torch.load(..., weights_only=True)``, which
    cannot unpickle SchNetPack checkpoints that embed custom model classes.
    Newer PyTorch Lightning versions expose ``weights_only`` on ``fit``; pass
    it only when supported (same pattern as ``load_task_from_checkpoint``).
    """
    kwargs = {}
    if "weights_only" in inspect.signature(trainer.fit).parameters:
        kwargs["weights_only"] = False
    return kwargs


def load_task_from_checkpoint(task_cls: type, ckpt_path: str, **kwargs: Any):
    """
    Load a task (LightningModule) from a Lightning checkpoint, handling
    `weights_only` compatibility across PyTorch Lightning versions.

    With torch >= 2.6 the checkpoint must be loaded with `weights_only=False`,
    but the corresponding `load_from_checkpoint` argument only exists in newer
    PyTorch Lightning versions — on PL <= 2.5.x it would be routed into the
    hparams overrides and break the task constructor. This helper passes the
    argument only when the installed PL supports it.

    Args:
        task_cls: The LightningModule subclass (e.g. AtomisticTask) to load.
        ckpt_path: Path to the Lightning checkpoint.
        **kwargs: Additional arguments for `load_from_checkpoint`.

    Returns:
        The loaded task instance.
    """
    if "weights_only" in inspect.signature(task_cls.load_from_checkpoint).parameters:
        kwargs.setdefault("weights_only", False)
    return task_cls.load_from_checkpoint(ckpt_path, **kwargs)

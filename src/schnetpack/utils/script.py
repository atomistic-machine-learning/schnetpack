from typing import Union, Dict, Sequence

import pytorch_lightning as pl
import rich
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree


__all__ = ["log_hyperparameters", "print_config"]


def empty(*args, **kwargs):
    pass


def todict(config: Union[DictConfig, Dict]):
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    return config_dict


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


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "run",
        "globals",
        "data",
        "model",
        "task",
        "trainer",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(
        f":gear: Running with the following config:", style=style, guide_style=style
    )

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)

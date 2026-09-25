import functools
import os
import random
from collections.abc import Sequence

import numpy as np
import rich
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax
from rich.tree import Tree

__all__ = ["print_config", "seed_everything"]


def _global_rank() -> int:
    """Global process rank as set by the launcher (torchrun, SLURM, ...)."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def rank_zero_only(fn):
    """Call ``fn`` only on the process with global rank 0."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if _global_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped


def seed_everything(seed: int) -> None:
    """Seed the random number generators of python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def todict(config: DictConfig | dict):
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    return config_dict


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
        ":gear: Running with the following config:", style=style, guide_style=style
    )

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)

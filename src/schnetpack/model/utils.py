from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch.nn as nn

__all__ = ["train_mode"]

#: class-name fragments of modules whose forward differs between train and eval
_TRAIN_MODE_SENSITIVE = ("Dropout", "BatchNorm", "InstanceNorm")


def train_mode_sensitive_modules(model: nn.Module) -> list[str]:
    """Class names of the submodules whose forward differs between train and
    eval mode, seeing through TorchScript wrappers."""
    names = {
        getattr(module, "original_name", type(module).__name__)
        for module in model.modules()
    }
    return sorted(
        name
        for name in names
        if any(fragment in name for fragment in _TRAIN_MODE_SENSITIVE)
    )


@contextlib.contextmanager
def train_mode(model: nn.Module) -> Generator[nn.Module, None, None]:
    """Run ``model`` in train mode, then restore its mode.

    SchNetPack's ``Forces`` differentiates with ``create_graph=self.training``:
    in eval mode the forces come back without a graph, and no Hessian-vector
    product can be taken through them (ADR-0010 §5).
    """
    was_training = model.training
    model.train(True)
    try:
        yield model
    finally:
        model.train(was_training)

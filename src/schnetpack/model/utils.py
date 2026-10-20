from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch.nn as nn

from schnetpack import properties

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


def forces_need_train_mode(model: nn.Module) -> bool:
    """Whether ``model``'s forces keep their graph only in train mode.

    Forces taken as a position derivative (``Forces``, ``MaceModel``) do: the
    model then lists the positions among its ``required_derivatives``. Forces
    predicted directly are an ordinary output and keep their graph in eval mode
    too. A model that lists no ``required_derivatives`` at all is assumed to
    need train mode.
    """
    derivatives = getattr(model, "required_derivatives", None)
    return derivatives is None or properties.R in derivatives


@contextlib.contextmanager
def train_mode(model: nn.Module) -> Generator[nn.Module, None, None]:
    """Run ``model`` in train mode, then restore its mode.

    SchNetPack's ``Forces`` differentiates with ``create_graph=self.training``:
    in eval mode the forces come back without a graph, and no curvature can be
    taken through them. :func:`forces_need_train_mode` says
    whether a model needs it.
    """
    was_training = model.training
    model.train(True)
    try:
        yield model
    finally:
        model.train(was_training)

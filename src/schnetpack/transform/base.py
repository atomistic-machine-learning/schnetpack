from typing import Optional, Dict

import torch
import torch.nn as nn

import schnetpack as spk

__all__ = [
    "Transform",
    "TransformException",
]


class TransformException(Exception):
    pass


class Transform(nn.Module):
    """
    Base class for all transforms.
    The base class ensures that the reference to the data and datamodule attributes are
    initialized.
    Transforms can be used as pre- or post-processing layers.
    They can also be used for other parts of a model, that need to be
    initialized based on data.

    To implement a new transform, override the forward method. Preprocessors are applied
    to single examples, while postprocessors operate on batches. All transforms should
    return a modified `inputs` dictionary.

    """

    def datamodule(self, value):
        """
        Extract all required information from data module automatically when using
        PyTorch Lightning integration. The transform should also implement a way to
        set these things manually, to make it usable independent of PL.

        Do not store the datamodule, as this does not work with torchscript conversion!
        """
        pass

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def teardown(self):
        pass

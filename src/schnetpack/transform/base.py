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
    The base class ensures that the reference to the data and datamodule attributes are initialized.

    To implement a new pre/post-processor, override the forward method. Preprocessors are applied
    to single examples.
    Post-processors are applied to batches and should return the transformed `results`. If a transform
    should be able to serve as both pre- and post-processor, use the `mode` attribute to process and
    return the respective arguments.
    """

    def datamodule(self, value):
        """
        Extract all required information from data module.

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

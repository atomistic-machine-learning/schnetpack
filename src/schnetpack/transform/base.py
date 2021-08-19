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
    to single examples and should ignore the `results` parameter and return the transformed `inputs`.
    Post-processors are applied to batches and should return the transformed `results`. If a transform
    should be able to serve as both pre- and post-processor, use the `mode` attribute to process and
    return the respective arguments.
    """

    def __init__(self):
        self.mode: str = "pre"
        super().__init__()

    def datamodule(self, value):
        """
        Extract all required information from data module.

        Do not store the datamodule, as this does not work with torchscript conversion!
        """
        pass

    def preprocessor(self):
        if not self.is_preprocessor:
            raise TransformException(
                f"Transform of type {type(self)} is not a preprocessor (is_preprocessor=False)!"
            )
        self.mode = "pre"

    def postprocessor(self):
        if not self.is_postprocessor:
            raise TransformException(
                f"Transform of type {type(self)} is not a post (is_postprocessor=False)!"
            )
        self.mode = "post"

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.mode == "post":
            return results
        else:
            return inputs

    def teardown(self):
        pass

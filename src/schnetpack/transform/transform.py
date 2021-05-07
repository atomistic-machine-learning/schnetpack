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
    Base class for all transforms. Only applied to single structures, not batches.
    The base class ensures that the reference to the data and datamodule attributes are initialized.

    To implement a new pre/post-processor, override the forward method. Preprocessors should ignore
    the `results` parameter and return the transformed `inputs`. Post-processors should return the
    transformed `results`. If a transform should be able to serve as both pre- and post-processor,
    use the `mode` attribute to process and return the respective arguments.
    """

    data: Optional["spk.data.BaseAtomsData"]
    datamodule: Optional["spk.data.AtomsDataModule"]

    def __init__(self):
        self._datamodule = None
        self._data = None
        self.mode = None
        super().__init__()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def datamodule(self):
        return self._datamodule

    @datamodule.setter
    def datamodule(self, value):
        self._datamodule = value

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

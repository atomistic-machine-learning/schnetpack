from typing import Dict

import torch
import torch.nn as nn

# import schnetpack as spk

__all__ = [
    "Transform",
    "TransformException",
]


class TransformException(Exception):
    pass


class Transform(nn.Module):
    """
    Base class for all transforms.
    Transforms can be used as pre- or post-processing layers.
    They can also be used for other parts of a model, that need to be
    initialized based on data.

    To implement a new transform, override the forward method. Preprocessors are applied
    to single examples, while postprocessors operate on batches. All transforms should
    return a modified `inputs` dictionary.

    Transforms that require training statistics override ``initialize``, the
    single initialization hook. It is called with a stats source both for
    data-pipeline transforms (during datamodule setup) and for model
    postprocessors (via AtomisticModel.initialize_transforms). Do not store
    the stats source, as this does not work with torchscript conversion!
    """

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def teardown(self):
        pass

    def initialize(self, stats=None) -> None:
        """
        Initialization hook for transforms that require training statistics.

        Args:
            stats: A stats source — any object providing ``get_stats`` and
                ``get_atomrefs`` (e.g. the datamodule or its stats provider).
        """
        return

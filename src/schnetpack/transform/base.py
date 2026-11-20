import torch
import torch.nn as nn

from schnetpack.model_format import register

# import schnetpack as spk

__all__ = [
    "Transform",
    "TransformException",
]


class TransformException(Exception):
    pass


@register
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

    #: Whether the transform changes positions or cells (noise, augmentation). A
    #: neighbor list is only valid when it is built after every such transform, and
    #: an :class:`~schnetpack.data.AtomsCache` refuses one among a dataset's
    #: transforms: it goes into the loader's ``batch_transforms``.
    moves_atoms: bool = False

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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

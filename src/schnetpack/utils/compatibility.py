import torch
import warnings
from typing import Any, Union


__all__ = ["load_model"]


def load_model(
    model_path: str, device: Union[torch.device, str] = "cpu", **kwargs: Any
) -> torch.nn.Module:
    """
    Load a SchNetPack model from a Torch file, enabling compatibility with models trained using earlier versions of
    SchNetPack. This function imports the old model and automatically updates it to the format used in the current
    SchNetPack version. To ensure proper functionality, the Torch model object must include a version tag, such as
    spk_version="2.0.4".

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device or str): Device on which to load the model. Defaults to "cpu".
        **kwargs (Any): Additional keyword arguments for `torch.load`.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = torch.load(model_path, map_location=device, weights_only=False, **kwargs)

    # convert old models to 2.0.4 format
    if not hasattr(model, "spk_version"):
        # make warning that model has no version information
        warnings.warn(
            "Model was saved without version information. Conversion to current version may fail."
        )
        model.spk_version = "2.0.4"

    # convert 2.0.4 models to 2.1.0 format
    if model.spk_version == "2.0.4":
        if not hasattr(model.representation, "electronic_embeddings"):
            model.representation.electronic_embeddings = []
        model.spk_version = "2.1.0"

    # convert 2.1.0 models to 2.1.1 format
    if model.spk_version == "2.1.0":
        # no conversion needed
        model.spk_version = "2.1.1"

    # convert 2.1.1 models to 2.2.0 format
    if model.spk_version == "2.1.1":
        # no conversion needed
        model.spk_version = "2.2.0"

    return model

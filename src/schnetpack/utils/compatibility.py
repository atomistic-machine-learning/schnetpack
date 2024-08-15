import torch
import warnings


__all__ = ["load_model"]


def load_model(model_path, device="cpu", **kwargs):
    """
    Load a SchNetPack model from a Torch file, enabling compatibility with models trained using earlier versions of
    SchNetPack. This function imports the old model and automatically updates it to the format used in the current
    SchNetPack version. To ensure proper functionality, the Torch model object must include a version tag, such as
    spk_version="2.0.4".

    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device on which the model should be loaded.
        **kwargs: Additional arguments for the model loading.

    Returns:
        torch.nn.Module: Loaded model.
    """

    def _convert_from_older(model):
        if not hasattr(model.representation, "electronic_embeddings"):
            model.representation.electronic_embeddings = []
        return model

    def _convert_from_v2_0_4(model):
        if not hasattr(model.representation, "electronic_embeddings"):
            model.representation.electronic_embeddings = []
        return model

    model = torch.load(model_path, map_location=device, **kwargs)
    if hasattr(model, "spk_version"):
        if model.spk_version == "2.0.4":
            model = _convert_from_v2_0_4(model)
    else:
        # make warning that model has no version information
        warnings.warn(
            "Model was saved without version information. Conversion to current version may fail."
        )
        model = _convert_from_older(model)

    return model

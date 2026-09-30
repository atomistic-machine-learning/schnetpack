import importlib
import sys
import warnings
from typing import Any

import torch

__all__ = ["load_model"]

# Modules that moved in the 3.0 reorganization, old dotted path -> new one.
# Pickled models store the module path of every class they contain, so a 2.x
# model file still refers to the paths on the left and cannot be unpickled
# without them. The package rows are needed as well, since importing
# ``schnetpack.representation.painn`` walks every level.
_LEGACY_MODULES = {
    "schnetpack.representation": "schnetpack.model.representation",
    "schnetpack.representation.painn": "schnetpack.model.representation.painn",
    "schnetpack.representation.schnet": "schnetpack.model.representation.schnet",
    "schnetpack.representation.field_schnet": "schnetpack.model.representation.field_schnet",
    "schnetpack.representation.so3net": "schnetpack.model.representation.so3net",
    "schnetpack.atomistic": "schnetpack.model.atomistic",
    "schnetpack.atomistic.atomwise": "schnetpack.model.atomistic.atomwise",
    "schnetpack.atomistic.response": "schnetpack.model.atomistic.response",
    "schnetpack.atomistic.distances": "schnetpack.model.atomistic.distances",
    "schnetpack.atomistic.aggregation": "schnetpack.model.atomistic.aggregation",
    "schnetpack.atomistic.electrostatic": "schnetpack.model.atomistic.electrostatic",
    "schnetpack.atomistic.external_fields": "schnetpack.model.atomistic.external_fields",
    "schnetpack.atomistic.nuclear_repulsion": "schnetpack.model.atomistic.nuclear_repulsion",
}


def _register_legacy_modules() -> None:
    """
    Make the pre-3.0 module paths importable again by aliasing them onto their
    new locations in ``sys.modules``, so that the pickle in an old model file
    resolves. Idempotent, and it never shadows a module that actually exists.
    """
    for old_name, new_name in _LEGACY_MODULES.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = importlib.import_module(new_name)


def load_model(
    model_path: str, device: torch.device | str = "cpu", **kwargs: Any
) -> torch.nn.Module:
    """
    Load a SchNetPack model from a Torch file, enabling compatibility with models trained using earlier versions of
    SchNetPack. This function imports the old model and automatically updates it to the format used in the current
    SchNetPack version. To ensure proper functionality, the Torch model object must include a version tag, such as
    spk_version="2.0.4".

    Modules that moved in the 3.0 reorganization (``schnetpack.representation.*`` and
    ``schnetpack.atomistic.*``, now below ``schnetpack.model``) are remapped before
    unpickling, see `_LEGACY_MODULES`.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device or str): Device on which to load the model. Defaults to "cpu".
        **kwargs (Any): Additional keyword arguments for `torch.load`.

    Returns:
        torch.nn.Module: Loaded model.
    """
    _register_legacy_modules()

    try:
        model = torch.load(
            model_path, map_location=device, weights_only=False, **kwargs
        )
    except (ModuleNotFoundError, AttributeError) as err:
        raise RuntimeError(
            f"Could not unpickle '{model_path}': {err}. The model refers to a class "
            "that no longer exists under that name, most likely because it was saved "
            "with an older SchNetPack version. If the class still exists elsewhere, "
            "add its old and new module path to `_LEGACY_MODULES` in "
            "schnetpack.utils.compatibility."
        ) from err

    # convert old models to 2.0.4 format
    if not hasattr(model, "spk_version"):
        # make warning that model has no version information
        warnings.warn(
            "Model was saved without version information. Conversion to current version may fail.",
            stacklevel=2,
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

    # convert 2.2.0 models to 3.0.0 format
    if model.spk_version == "2.2.0":
        from schnetpack.model.representation import PaiNN

        # `norm_epsilon` is a plain attribute, so old PaiNN instances lack it
        representation = getattr(model, "representation", None)
        if isinstance(representation, PaiNN) and not hasattr(
            representation, "norm_epsilon"
        ):
            representation.norm_epsilon = 0.0
        model.spk_version = "3.0.0"

    return model

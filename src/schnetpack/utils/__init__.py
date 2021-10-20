import importlib
import torch
from typing import Type, Union, List

from schnetpack import properties as spk_properties


def int2precision(precision: Union[int, torch.dtype]):
    """
    Get torch floating point precision from integer.
    If an instance of torch.dtype is passed, it is returned automatically.

    Args:
        precision (int, torch.dtype): Target precision.

    Returns:
        torch.dtupe: Floating point precision.
    """
    if isinstance(precision, torch.dtype):
        return precision
    else:
        try:
            return getattr(torch, f"float{precision}")
        except AttributeError:
            raise AttributeError(f"Unknown float precision {precision}")


def str2class(class_path: str) -> Type:
    """
    Obtain a class type from a string

    Args:
        class_path: module path to class, e.g. ``module.submodule.classname``

    Returns:
        class type
    """
    class_path = class_path.split(".")
    class_name = class_path[-1]
    module_name = ".".join(class_path[:-1])
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls


def required_fields_from_properties(properties: List[str]) -> List[str]:
    """
    Determine required external fields based on the response properties to be computed.

    Args:
        properties (list(str)): List of response properties for which external fields should be determined.

    Returns:
        list(str): List of required external fields.
    """
    required_fields = set()

    for p in properties:
        if p in spk_properties.required_external_fields:
            required_fields.update(spk_properties.required_external_fields[p])

    required_fields = list(required_fields)

    return required_fields

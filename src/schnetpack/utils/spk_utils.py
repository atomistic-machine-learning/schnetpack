import json
import logging
from argparse import Namespace

import numpy as np
import torch


__all__ = [
    "set_random_seed",
    "count_params",
    "to_json",
    "read_from_json",
    "DeprecationHelper",
]


def set_random_seed(seed):
    """
    This function sets the random seed (if given) or creates one for torch and numpy random state initialization

    Args:
        seed (int, optional): if seed not present, it is generated based on time
    """
    import time
    import numpy as np

    # 1) if seed not present, generate based on time
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
            ((seed & 0xFF000000) >> 24)
            + ((seed & 0x00FF0000) >> 8)
            + ((seed & 0x0000FF00) << 8)
            + ((seed & 0x000000FF) << 24)
        )
    # 2) Set seed for numpy (e.g. splitting)
    np.random.seed(seed)
    # 3) Set seed for torch (manual_seed now seeds all CUDA devices automatically)
    torch.manual_seed(seed)
    logging.info("Random state initialized with seed {:<10d}".format(seed))


def count_params(model):
    """
    This function takes a model as an input and returns the number of
    trainable parameters.

    Args:collect
        model (AtomisticModel): model for which you want to count
                                the trainable parameters

    Returns:
        params (int): number of trainable parameters for the model
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def to_json(jsonpath, argparse_dict):
    """
    This function creates a .json file as a copy of argparse_dict

    Args:
        jsonpath (str): path to the .json file
        argparse_dict (dict): dictionary containing arguments from argument parser
    """
    with open(jsonpath, "w") as fp:
        json.dump(argparse_dict, fp, sort_keys=True, indent=4)


def read_from_json(jsonpath):
    """
    This function reads args from a .json file and returns the content as a namespace dict

    Args:
        jsonpath (str): path to the .json file

    Returns:
        namespace_dict (Namespace): namespace object build from the dict stored into the given .json file.
    """
    with open(jsonpath) as handle:
        dict = json.loads(handle.read())
        namespace_dict = Namespace(**dict)
    return namespace_dict


class DeprecationHelper(object):
    def __init__(self, new_target, old_name):
        self.new_target = new_target
        self.old_name = old_name

    def _warn(self):
        from warnings import warn

        warn(
            self.old_name
            + "is deprecated, use "
            + self.new_target.__class__.__name__
            + " instead",
            DeprecationWarning,
        )

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)

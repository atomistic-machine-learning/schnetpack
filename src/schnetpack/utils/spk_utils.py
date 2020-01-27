import json
import logging
from argparse import Namespace

import numpy as np
import torch
from ase.db import connect
from base64 import b64decode
from tqdm import tqdm

__all__ = [
    "set_random_seed",
    "count_params",
    "to_json",
    "read_from_json",
    "DeprecationHelper",
    "load_model",
    "read_deprecated_database",
    "activate_stress_computation",
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


def load_model(model_path, map_location=None):
    """
    Wrapper function for `for safely loading models where certain new attributes of the model class are not present.
    E.g. "requires_stress" for computing the stress tensor.

    Args:
        model_path (str): Path to the model file.
        map_location (torch.device): Device where the model should be loaded to.

    Returns:
        :class:`schnetpack.atomistic.AtomisticModel`: Loaded SchNetPack model.

    """
    model = torch.load(model_path, map_location=map_location)

    # Check for data parallel models
    if hasattr(model, "module"):
        model_module = model.module
        output_modules = model.module.output_modules
    else:
        model_module = model
        output_modules = model.output_modules

    # Set stress tensor attribute if not present
    if not hasattr(model_module, "requires_stress"):
        model_module.requires_stress = False
        for module in output_modules:
            module.stress = None

    return model


def read_deprecated_database(db_path):
    """
    Read all atoms and properties from deprecated ase databases.

    Args:
        db_path (str): Path to deprecated database

    Returns:
        atoms (list): All atoms objects of the database.
        properties (list): All property dictionaries of the database.

    """
    with connect(db_path) as conn:
        db_size = conn.count()
    atoms = []
    properties = []

    for idx in tqdm(range(1, db_size + 1), "Reading deprecated database"):
        with connect(db_path) as conn:
            row = conn.get(idx)

        at = row.toatoms()
        pnames = [pname for pname in row.data.keys() if not pname.startswith("_")]
        props = {}
        for pname in pnames:
            try:
                shape = row.data["_shape_" + pname]
                dtype = row.data["_dtype_" + pname]
                prop = np.frombuffer(b64decode(row.data[pname]), dtype=dtype)
                prop = prop.reshape(shape)
            except:
                # fallback for properties stored directly
                # in the row
                if pname in row:
                    prop = row[pname]
                else:
                    prop = row.data[pname]

                try:
                    prop.shape
                except AttributeError as e:
                    prop = np.array([prop], dtype=np.float32)
            props[pname] = prop

        atoms.append(at)
        properties.append(props)

    return atoms, properties


def activate_stress_computation(model, stress="stress"):
    """
    Utility function to activate the computation of the stress tensor for a model not trained explicitly on
    this property. It is recommended to at least have used forces during training when switching on the stress.
    Moreover, now proper crystal cell (volume > 0) needs to be specified for the molecules.

    Args:
        model (schnetpack.atomistic.AtomisticModel): SchNetPack model for which computation of the stress tensor
                                                    should be activated.
        stress (str): Designated name of the stress tensor property used in the model output.
    """
    # Check for data parallel models
    if hasattr(model, "module"):
        model_module = model.module
        output_modules = model.module.output_modules
    else:
        model_module = model
        output_modules = model.output_modules

    # Set stress tensor attribute if not present
    if hasattr(model_module, "requires_stress"):
        model_module.requires_stress = True
        for module in output_modules:
            if hasattr(module, "stress"):
                module.stress = stress

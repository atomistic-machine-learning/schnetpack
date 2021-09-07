from .md_config import *
from .normal_model_transformation import *
from .thermostat_utils import *

from typing import Optional
import torch
import torch.nn as nn


class UninitializedMixin(nn.modules.lazy.LazyModuleMixin):
    """
    Custom mixin for lazy initialization of buffers used in the MD system and simulation hooks.
    This can be used to add buffers with a certain dtype in an uninitialized state.
    """

    def register_uninitialized_buffer(
        self, name: str, dtype: Optional[torch.dtype] = None
    ):
        """
        Register an uninitialized buffer with the requested dtype. This can be used to reserve variable which are not
        known at the initialization of `schnetpack.md.System` and simulation hooks.

        Args:
            name (str): Name of the uninitialized buffer to register.
            dtype (torch.dtype): If specified, buffer will be set to requested dtype. If None is given, this will
                                 default to float64 type.
        """
        if dtype is None:
            dtype = torch.float64

        self.register_buffer(name, nn.parameter.UninitializedBuffer(dtype=dtype))

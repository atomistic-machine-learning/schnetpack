from typing import Dict
from typing import Dict, Optional
from schnetpack.utils import as_dtype

import torch

from .base import Transform

__all__ = ["CastMap", "CastTo32", "CastTo64"]


class CastMap(Transform):
    """
    Cast all inputs according to type map.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(self, type_map: Dict[str, str]):
        """
        Args:
            type_map: dict with source_type: target_type (as strings)
        """
        super().__init__()
        self.type_map = type_map

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        for k, v in inputs.items():
            vdtype = str(v.dtype).split(".")[-1]
            if vdtype in self.type_map:
                inputs[k] = v.to(dtype=as_dtype(self.type_map[vdtype]))
        return inputs


class CastTo32(CastMap):
    """Cast all float64 tensors to float32"""

    def __init__(self):
        super().__init__(type_map={"float64": "float32"})


class CastTo64(CastMap):
    """Cast all float32 tensors to float64"""

    def __init__(self):
        super().__init__(type_map={"float32": "float64"})

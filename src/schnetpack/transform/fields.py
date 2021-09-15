import torch

from typing import Optional, Dict, List

from .base import Transform, TransformException
from schnetpack import properties
from schnetpack.utils import required_fields_from_properties

__all__ = ["ExternalFields"]


class ExternalFields(Transform):
    """
    Transform for adding external fields and nuclear magnetic moments to a molecule if not present in the data.
    This is typically used if the external fields are assumed to be 0, but response properties should be modeled.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    implemented_fields = [properties.electric_field, properties.magnetic_field]

    def __init__(
        self,
        required_fields: List[str] = [],
        response_properties: Optional[List[str]] = None,
    ):
        """

        Args:
            required_fields (list(str)): List of the required fields.
            response_properties (list(str), optional): List of response properties. If this is provided instead of the
                                                       `required_fields`, the necessary external fields are determined
                                                       based on the requested response properties.
        """
        super(ExternalFields, self).__init__()

        if response_properties is not None:
            required_fields = required_fields_from_properties(response_properties)

        for field in required_fields:
            if field not in self.implemented_fields:
                raise TransformException(
                    "{:s} not in implemented external fields.".format(field)
                )

        self.electric_field = properties.electric_field in required_fields
        self.magnetic_field = properties.magnetic_field in required_fields

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        results: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        dtype = inputs[properties.R].dtype
        device = inputs[properties.R].device

        # Set up electric field
        if self.electric_field:
            if properties.electric_field not in inputs:
                inputs[properties.electric_field] = torch.zeros(
                    (1, 3), dtype=dtype, device=device
                )

        # Set up magnetic field
        if self.magnetic_field:
            # basic field
            if properties.magnetic_field not in inputs:
                inputs[properties.magnetic_field] = torch.zeros(
                    (1, 3), dtype=dtype, device=device
                )

            # Set up nuclear magnetic moments
            if properties.nuclear_magnetic_moments not in inputs:
                inputs[properties.nuclear_magnetic_moments] = torch.zeros_like(
                    inputs[properties.R]
                )

        return inputs

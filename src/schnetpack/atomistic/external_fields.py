from typing import Dict, Optional, List

import torch
import torch.nn as nn

import schnetpack.properties as properties
from schnetpack.utils import required_fields_from_properties

__all__ = ["StaticExternalFields"]


class StaticExternalFields(nn.Module):
    """
    Input routine for setting up dummy external fields in response models.
    Checks if fields are present in input and sets dummy fields otherwise.

    Args:
        external_fields (list(str)): List of required external fields. Either this or the requested response
                                     properties needs to be specified.
        response_properties (list(str)): List of requested response properties. If this is not None, it is used to
                                         determine the required external fields.
    """

    def __init__(
        self,
        external_fields: List[str] = [],
        response_properties: Optional[List[str]] = None,
    ):
        super(StaticExternalFields, self).__init__()

        if response_properties is not None:
            external_fields = required_fields_from_properties(response_properties)

        self.external_fields: List[str] = list(set(external_fields))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        n_atoms = inputs[properties.n_atoms]
        n_molecules = n_atoms.shape[0]

        # Fields passed to interaction computation (cast to batch structure)
        for field in self.external_fields:
            # Store all fields in directory which will be returned for derivatives
            if field not in inputs:
                inputs[field] = torch.zeros(
                    n_molecules,
                    3,
                    device=n_atoms.device,
                    dtype=inputs[properties.R].dtype,
                    requires_grad=True,
                )

        # Initialize nuclear magnetic moments for magnetic fields
        if properties.magnetic_field in self.external_fields:
            if properties.nuclear_magnetic_moments not in inputs:
                inputs[properties.nuclear_magnetic_moments] = torch.zeros_like(
                    inputs[properties.R], requires_grad=True
                )

        return inputs

import torch

from schnetpack.transform.base import Transform
from schnetpack import properties

from typing import Dict, List

__all__ = ["SplitShielding"]


class SplitShielding(Transform):
    """
    Transform for splitting shielding tensors by atom types.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        shielding_key: str,
        atomic_numbers: List[int],
    ):
        """
        Args:
            shielding_key (str): name of the shielding tensor in the model inputs.
            atomic_numbers (list(int)): list of atomic numbers used to split the shielding tensor.
        """
        super(SplitShielding, self).__init__()

        self.shielding_key = shielding_key
        self.atomic_numbers = atomic_numbers

        self.model_outputs = [
            "{:s}_{:d}".format(self.shielding_key, atomic_number)
            for atomic_number in self.atomic_numbers
        ]

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        shielding = inputs[self.shielding_key]

        split_shielding = {}
        for atomic_number in self.atomic_numbers:
            atomic_key = "{:s}_{:d}".format(self.shielding_key, atomic_number)
            split_shielding[atomic_key] = shielding[
                inputs[properties.Z] == atomic_number, :, :
            ]

        inputs.update(split_shielding)

        return inputs

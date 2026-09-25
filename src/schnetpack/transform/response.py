import torch

from schnetpack import properties
from schnetpack.transform.base import Transform

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
        atomic_numbers: list[int],
    ):
        """
        Args:
            shielding_key (str): name of the shielding tensor in the model inputs.
            atomic_numbers (list(int)): list of atomic numbers used to split the shielding tensor.
        """
        super().__init__()

        self.shielding_key = shielding_key
        self.atomic_numbers = atomic_numbers

        self.model_outputs = [
            f"{self.shielding_key:s}_{atomic_number:d}"
            for atomic_number in self.atomic_numbers
        ]

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        shielding = inputs[self.shielding_key]

        split_shielding = {}
        for atomic_number in self.atomic_numbers:
            atomic_key = f"{self.shielding_key:s}_{atomic_number:d}"
            split_shielding[atomic_key] = shielding[
                inputs[properties.Z] == atomic_number, :, :
            ]

        inputs.update(split_shielding)

        return inputs

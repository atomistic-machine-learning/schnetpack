import torch
import torch.nn as nn

from typing import Dict, List

__all__ = ["Aggregation"]


class Aggregation(nn.Module):
    """
    Aggregate predictions into a single output variable.

    Args:
        keys (list(str)): List of properties to be added.
        output_key (str): Name of new property in output.
    """

    def __init__(self, keys: List[str], output_key: str = "y"):
        super(Aggregation, self).__init__()

        self.keys: List[str] = list(keys)
        self.output_key = output_key

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = {}
        energy = torch.stack([inputs[key] for key in self.keys]).sum(0)
        result[self.output_key] = energy
        return result

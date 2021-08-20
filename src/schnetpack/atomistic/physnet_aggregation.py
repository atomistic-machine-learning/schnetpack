import torch
import torch.nn as nn
import schnetpack.properties as structure

from typing import Sequence, Union, Callable, Dict, Optional, List

__all__ = ["Aggregation"]

class Aggregation(nn.Module):
    
    def __init__(
            self, 
            keys : List[str], 
            output_key: str = "y"
            
        ):
        
        super(Aggregation, self).__init__()
   
        self.keys = keys
        self.output_key = output_key
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        result = {}
        energy = torch.stack([inputs[key] for key in self.keys]).sum(0)
        result[self.output_key] = energy
        
        return result
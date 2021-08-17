import torch
import torch.nn as nn
import schnetpack.properties as structure

from typing import Sequence, Union, Callable, Dict, Optional

class Aggregation(nn.Module):
    
    def __init__(
            self, 
            calc_electrostatic: bool = True,
            calc_zbl: bool = True,
            calc_dispersion: bool = True,
            electrostatic_key: str = structure.electrostatic,
            zbl_key: str = structure.zbl,
            dispersion_key: str = structure.dispersion,
            aggregate_key: str = structure.physnet_aggregate
        ):
        
        super(Aggregation, self).__init__()
   
        self.electrostatic_key = electrostatic_key
        self.zbl_key = zbl_key
        self.dispersion_key = dispersion_key
        self.aggregate_key = aggregate_key
        self.calc_electrostatic = calc_electrostatic
        self.calc_zbl = calc_zbl
        self.calc_dispersion = calc_dispersion
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        result = {}
        energy = []
        
        if self.calc_electrostatic:
            energy.append(inputs[self.electrostatic_key])
        if self.calc_zbl:
            energy.append(inputs[self.zbl_key])
        if self.calc_dispersion:
            energy.append(inputs[self.dispersion_key])
            
        energy = torch.stack(energy).sum(0)
        result[self.aggregate_key] = energy.unsqueeze(-1)
        
        return result
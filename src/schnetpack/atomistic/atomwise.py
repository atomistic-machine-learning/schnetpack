from typing import Sequence, Union, Callable, Dict, Optional

import torch
import torch.nn as nn

import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.atomistic.physnet_energy import PhysNetEnergy
import schnetpack.nn as snn


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
        self,
        output: str,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: str = "sum",
        custom_outnet: Callable = None,
        module_dim = False,
        calc_electrostatic: bool = False
        calc_zbl: bool = False
        calc_dispersion: bool = False
        outnet_input: Union[str, Sequence[str]] = "scalar_representation",
        electrostatic_key: str = structure.electrostatic,
        zbl_key: str = structure.zbl,
        dispersion_key = structure.dispersion   
    ):
        """
        Args:
            output: the key under which the result should be stored
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            aggregation_mode: one of {sum, avg} (default: sum)
            custom_outnet: Network used for atomistic outputs. Takes schnetpack input
                dictionary as input. Output is not normalized. If set to None,
                a pyramidal network is generated automatically.
            outnet_inputs: input dict entries to pass to outnet.
        """
        super(Atomwise, self).__init__()
        self.output = output
        self.n_out = n_out
        
        self.electrostatic_key = electrostatic_key
        self.zbl_key = zbl_key
        self.dispersion_key = dispersion_key
        
        self.calc_electrostatic = calc_electrostatic
        self.calc_zbl = calc_zbl
        self.calc_dispersion = calc_dispersion
        
        

        #build output network
        self.outnet = custom_outnet or spk.nn.MLP(
            n_in=n_in, n_out=n_out, n_layers=2, activation=spk.nn.shifted_softplus
        )
        
#         # Getting moved to config
        if module_dim:
            self.outnet.weight.data = torch.nn.Parameter(torch.zeros(2,n_in))
            self.outnet.bias.data.fill_(0.)
        
        self.outnet_input = outnet_input

        self.aggregation_mode = aggregation_mode
        
        self.module_dim = module_dim
        
        #self.physnet_energy = PhysNetEnergy()
            

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        if self.module_dim:
            inputs[self.outnet_input] = inputs[self.outnet_input].sum(0)

        # predict atomwise contributions
        yi = inputs[self.outnet_input]
        yi = self.outnet(inputs[self.outnet_input])
        
        if self.calc_electrostatic:
            yi += inputs[self.electrostatic_key]
        if self.calc_zbl:
            yi += inputs[self.zbl_key]
        if self.calc_dispersion:
            yi += inputs[self.dispersion_key]
        #yi = self.physnet_energy(yi, inputs)
        
        if self.aggregation_mode == "avg":
            yi = yi / inputs[structure.n_atoms][:, None]

        # aggregate
        idx_m = inputs[structure.idx_m]
        maxm = int(idx_m[-1]) + 1
        tmp = torch.zeros((maxm, self.n_out), dtype=yi.dtype, device=yi.device)
        y = tmp.index_add(0, idx_m, yi)
        y = torch.squeeze(y, -1)

        return {self.output: y}

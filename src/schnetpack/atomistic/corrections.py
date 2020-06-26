import torch
import torch.nn as nn
import schnetpack as spk


__all__ = ["ElectrostaticEnergy"]


class Correction(nn.Module):
    def __init__(self):
        super(Correction, self).__init__()

    def forward(self, inputs):
        pass


class ElectrostaticEnergy(nn.Module):
    def __init__(self, cuton, cutoff, ke=14.399645351950548, lr_cutoff=None):
        super(ElectrostaticEnergy, self).__init__()
        self.ke = ke
        self.cuton = cuton
        self.cutoff = cutoff
        self.lr_cutoff = lr_cutoff
        # these are constants for when a lr_cutoff is used
        if lr_cutoff is not None:
            self.cut_rconstant = lr_cutoff ** 15 / (lr_cutoff ** 16 + cuton ** 16) ** (
                17 / 16
            )
            self.cut_constant = 1 / (cuton ** 16 + lr_cutoff ** 16) ** (
                1 / 16
            ) + lr_cutoff ** 16 / (lr_cutoff ** 16 + cuton ** 16) ** (17 / 16)

    def forward(self, inputs, atomwise_predictions):
        # get properties
        qi = atomwise_predictions["qi"]
        r_ij = inputs["distances"]
        neighbors = inputs[spk.Properties.neighbors]
        neighbor_mask = inputs[spk.Properties.neighbor_mask]

        # todo: double check this!
        # get qi*qj matrix
        q_ij = qi * qi.transpose(1, 2)
        # remove diagonal elements
        q_ij = torch.gather(q_ij, -1, neighbors) * neighbor_mask

        # compute switch factors
        f = spk.nn.switch_function(r_ij, self.cuton, self.cutoff)

        # compute damped and coulomb components
        if self.lr_cutoff is None:
            coulomb = torch.where(neighbor_mask!=0., 1 / r_ij, torch.zeros_like(r_ij))
            damped = torch.where(
                neighbor_mask!=0.,
                1 / (r_ij ** 16 + self.cuton ** 16) ** (1 / 16),
                torch.zeros_like(r_ij)
            )
        else:
            coulomb = torch.where(
                r_ij < self.lr_cutoff,
                1.0 / r_ij + r_ij / self.lr_cutoff ** 2 - 2.0 / self.lr_cutoff,
                torch.zeros_like(r_ij),
            )
            damped = (
                1 / (r_ij ** 16 + self.cuton ** 16) ** (1 / 16)
                + (1 - f) * self.cut_rconstant * r_ij
                - self.cut_constant
            )

        # return sum over all atoms i and neighbors j
        return torch.sum(
            self.ke / 2 * q_ij * (f * damped + (1 - f) * coulomb * neighbor_mask),
            [-1, -2]
        ).unsqueeze(-1)

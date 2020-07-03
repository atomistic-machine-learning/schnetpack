import torch
import torch.nn as nn

__all__ = ["build_mse_loss"]


class LossFnError(Exception):
    pass


def build_mse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            diff = batch[prop] - result[prop]
            diff = diff ** 2
            err_sq = factor * torch.mean(diff)
            loss += err_sq
        return loss

    return loss_fn


class PhysNetLoss(nn.Module):
    def __init__(
        self,
        energy_factor=1.0,
        forces_factor=0.0,
        charge_factor=0.0,
        dipole_factor=0.0,
        nh_penalty_factor=1e-2,
    ):
        super(PhysNetLoss, self).__init__()

        # store attributes
        self.energy_factor = energy_factor
        self.forces_factor = forces_factor
        self.charge_factor = charge_factor
        self.dipole_factor = dipole_factor
        self.nh_penalty_factor = nh_penalty_factor

    def forward(
        self,
        E=None,
        E_ref=None,
        F_i=None,
        F_i_ref=None,
        q_i=None,
        Q_ref=None,
        p_ref=None,
        E_i=None,
        r_i=None,
        atom_mask=None,
    ):
        N = atom_mask.sum(1)

        loss = 0.0

        # energy contribution
        if self.energy_factor > 0.0:
            energy_loss = torch.mean(torch.abs(E - E_ref))
            loss += self.energy_factor * energy_loss

        # forces contribution
        if self.forces_factor > 0.0:
            force_loss = torch.sum(1 / (3 * N) * torch.sum(F_i - F_i_ref, 1))
            loss += self.forces_factor * force_loss

        # charges contribution
        if self.charges_factor > 0.0:
            charge_loss = torch.sum(torch.sum(q_i, -1), Q_ref)
            loss += self.charge_factor * charge_loss

        # dipole contribution
        if self.dipole_factor > 0.0:
            dipole_loss = torch.mean(torch.sum(q_i * r_i, -1) - p_ref)

        # non-hierarchicality penalty
        # todo: not sure...

        return loss

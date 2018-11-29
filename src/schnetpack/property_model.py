import torch
import torch.nn as nn

from schnet_transfer.atomistic import Polarizability
from schnetpack.atomistic import Energy, DipoleMoment


class ModelError(Exception):
    pass


class Properties:
    energy = 'energy'
    forces = 'forces'
    dipole_moment = 'dipole_moment'
    total_dipole_moment = 'total_dipole_moment'
    polarizability = 'polarizability'
    iso_polarizability = 'iso_polarizability'
    at_polarizability = 'at_polarizability'
    charges = 'charges'
    energy_contributions = 'energy_contributions'


class PropertyModel(nn.Module):

    def __init__(self, n_in, properties, mean, stddev, atomrefs,
                 cutoff_network, cutoff):
        super(PropertyModel, self).__init__()

        self.n_in = n_in
        self.properties = properties

        self.need_energy = Properties.energy in properties
        self.need_forces = Properties.forces in properties
        self.need_dipole = Properties.dipole_moment in properties
        self.need_total_dipole = Properties.total_dipole_moment in properties
        self.need_polarizability = Properties.polarizability in properties
        self.need_at_polarizability = Properties.at_polarizability in properties
        self.need_iso_polarizability = Properties.iso_polarizability in properties
        self.need_energy_contributions = \
            Properties.energy_contributions in properties
        self.requires_dr = self.need_forces

        # if self.need_total_dipole and self.need_dipole:
        #     raise ModelError("Only one of dipole_moment and " + \
        #                      "total_dipole_moment can be specified!")

        self.cutoff_network = cutoff_network(cutoff)

        outputs = {}
        self.bias = {}
        if self.need_energy or self.need_forces:
            mu = torch.tensor(mean[Properties.energy])
            std = torch.tensor(stddev[Properties.energy])
            try:
                atomref = atomrefs[Properties.energy]
            except:
                atomref = None

            energy_module = Energy(
                n_in, aggregation_mode='sum', return_force=self.need_forces,
                return_contributions=self.need_energy_contributions,
                mean=mu, stddev=std, atomref=atomref, create_graph=True)
            outputs[Properties.energy] = energy_module
            self.bias[Properties.energy] = energy_module.out_net[1].out_net[
                1].bias
        if self.need_dipole or self.need_total_dipole:
            dipole_module = DipoleMoment(
                n_in, predict_magnitude=self.need_total_dipole,
                return_charges=True,
            )
            if self.need_dipole:
                outputs[Properties.dipole_moment] = dipole_module
            else:
                outputs[Properties.total_dipole_moment] = dipole_module
        if self.need_polarizability or self.need_iso_polarizability:
            polar_module = Polarizability(
                n_in, return_isotropic=self.need_iso_polarizability,
                cutoff_network=self.cutoff_network)
            outputs[Properties.polarizability] = polar_module

        self.output_dict = nn.ModuleDict(outputs)

    def forward(self, inputs):
        outs = {}
        for prop, mod in self.output_dict.items():
            o = mod(inputs)
            outs[prop] = o['y']
            if prop == Properties.energy and self.need_forces:
                outs[Properties.forces] = o['dydx']
            if prop == Properties.energy and self.need_energy_contributions:
                outs[Properties.energy_contributions] = o['yi']
            if prop in [Properties.dipole_moment,
                        Properties.total_dipole_moment]:
                outs[Properties.charges] = o['yi']
            if prop == Properties.polarizability:
                if self.need_iso_polarizability:
                    outs[Properties.iso_polarizability] = o['y_iso']
                if self.need_at_polarizability:
                    outs[Properties.at_polarizability] = o['y_i']
        return outs

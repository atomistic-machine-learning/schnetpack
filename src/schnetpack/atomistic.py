r"""
Classes for output modules.
"""

from collections import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.autograd import grad

import schnetpack.nn.activations
import schnetpack.nn.base
import schnetpack.nn.blocks
from schnetpack.data import Structure
import schnetpack.nn as L


class AtomisticModel(nn.Module):
    """
    Assembles an atomistic model from a representation module and one or more output modules.

    Returns either the predicted property or a dict (output_module -> property),
    depending on whether output_modules is a Module or an iterable.

    Args:
        representation(nn.Module): neural network that builds atom-wise features of shape batch x atoms x features
        output_modules(nn.OutputModule or iterable of nn.OutputModule): predicts desired property from representation


    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()

        self.representation = representation

        if isinstance(output_modules, Iterable):
            self.output_modules = nn.ModuleList(output_modules)
            self.requires_dr = False
            for o in self.output_modules:
                if o.requires_dr:
                    self.requires_dr = True
                    break
        else:
            self.output_modules = output_modules
            self.requires_dr = output_modules.requires_dr

    def forward(self, inputs):
        r"""
        predicts property

        """
        if self.requires_dr:
            inputs[Structure.R].requires_grad_()
        inputs["representation"] = self.representation(inputs)

        if isinstance(self.output_modules, nn.ModuleList):
            outs = {}
            for output_module in self.output_modules:
                outs[output_module] = output_module(inputs)
        else:
            outs = self.output_modules(inputs)

        return outs


class OutputModule(nn.Module):
    r"""
    Base class for output modules.

    Args:
        requires_dr (bool): specifies if the derivative of the ouput is required
    """

    def __init__(self, requires_dr=False):
        self.requires_dr = requires_dr
        super(OutputModule, self).__init__()

    def forward(self, inputs):
        r"""
        Should be overwritten
        """
        raise NotImplementedError


class Atomwise(OutputModule):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    Args:
        n_in (int): input dimension of representation (default: 128)
        n_out (int): output dimension of target property (default: 1)
        pool_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output network.
                                          If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
        return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
        requires_dr (bool): True, if derivative w.r.t. atom positions is required (default: False)
        mean (torch.FloatTensor): mean of property (default: None)
        stddev (torch.FloatTensor): standard deviation of property (default: None)
        atomref (torch.Tensor): reference single-atom properties. Expects an (max_z + 1) x 1 array where atomref[Z]
                                corresponds to the reference property of element Z. The value of atomref[0] must be
                                zero, as this corresponds to the reference property for for "mask" atoms
        max_z (int): only relevant only if train_embeddings is true.
                     Specifies maximal nuclear charge of atoms. (default: 100)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                           not normalized. If set to None (default), a pyramidal network is generated automatically.
        train_embeddings (bool): if set to true, atomref will be ignored and learned from data (default: None)

    Returns:
        tuple: prediction for property

        If return_contributions is true additionally returns atom-wise contributions.

        If requires_dr is true additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in=128,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        contribution_property=None,
        dr_property=None,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        max_z=100,
        outnet=None,
        train_embeddings=False,
        property="y"
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contribution_property = contribution_property
        self.dr_property = dr_property

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref[self.property].astype(np.float32)),
                freeze=not train_embeddings,
            )
        elif train_embeddings:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(np.zeros((max_z, 1), dtype=np.float32)),
                freeze=not train_embeddings,
            )
        else:
            self.atomref = None

        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # Make standardization separate
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        if aggregation_mode == "sum":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Structure.Z]
        atom_mask = inputs[Structure.atom_mask]

        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi, atom_mask)
        # add property
        result = {self.property: y}
        # add property contributions
        if self.contribution_property:
            result[self.contribution_property] = yi
        # add property derivative
        if self.dr_property:
            dy = -grad(
                result[self.property],
                inputs[Structure.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                # todo: check here
                retain_graph=True
            )[0]
            result[self.dr_property] = dy

        return result


class DipoleMoment(Atomwise):
    """
    Predicts latent partial charges and calculates dipole moment.

    Args:
        n_in (int): input dimension of representation
        n_layers (int): number of layers in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output network.
                                          If `None`, divide neurons by 2 in each layer. (default: none)
        activation (torch.Function): activation function for hidden nn (default: schnetpack.nn.activations.shifted_softplus)
        return_charges (bool): if True, latent atomic charges are returned as well (default: False)
        requires_dr (bool): set True, if derivative w.r.t. atom positions is required (default: False)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment instead of the vector (default: False)
        mean (torch.FloatTensor): mean of dipole (default: 0.0)
        stddev (torch.FloatTensor): stddev of dipole (default: 0.0)


    Returns:
        tuple: vector for the dipole moment

        If predict_magnitude is true returns the magnitude of the dipole moment instead of the vector

        If return_charges is true returns either vector or magnitude of the dipole moment, and latent atomic charges

        If requires_dr is true returns derivative w.r.t. atom positions
    """

    def __init__(
        self,
        n_in,
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        contribution_property=None,
        dr_property=None,
        outnet=None,
        predict_magnitude=False,
        mean=torch.FloatTensor([0.0]),
        stddev=torch.FloatTensor([1.0]),
    ):
        self.contribution_property = contribution_property
        self.predict_magnitude = predict_magnitude
        super(DipoleMoment, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_neurons,
            activation=activation,
            contribution_property=contribution_property,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
        )

    def forward(self, inputs):
        """
        predicts dipole moment
        """
        positions = inputs[Structure.R]
        atom_mask = inputs[Structure.atom_mask][:, :, None]

        charges = self.out_net(inputs) * atom_mask
        yi = positions * charges
        y = self.atom_pool(yi)

        if self.predict_magnitude:
            result = {"y": torch.norm(y, dim=1, keepdim=True)}
        else:
            result = {"y": y}

        if self.return_charges:
            result["yi"] = charges

        return result


class ElementalAtomwise(Atomwise):
    """
    Predicts properties in atom-wise fashion using a separate network for every chemical element of the central atom.
    Particularly useful for networks of Behler-Parrinello type.
    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=3,
        dr_property=None,
        create_graph=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden=50,
        activation=schnetpack.nn.activations.shifted_softplus,
        contribution_property=False,
        mean=None,
        stddev=None,
        atomref=None,
        max_z=100,
        property="y"
    ):
        outnet = schnetpack.nn.blocks.GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalAtomwise, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            None,
            activation,
            contribution_property,
            dr_property,
            create_graph,
            mean,
            stddev,
            atomref,
            max_z,
            outnet,
            property=property
        )


class Polarizability(Atomwise):
    def __init__(
        self,
        n_in,
        pool_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=L.shifted_softplus,
        return_isotropic=False,
        return_contributions=False,
        create_graph=False,
        outnet=None,
        cutoff_network=None,
    ):
        super(Polarizability, self).__init__(
            n_in,
            2,
            pool_mode,
            n_layers,
            n_neurons,
            activation,
            return_contributions,
            True,
            create_graph,
            None,
            None,
            None,
            100,
            outnet,
        )
        self.return_isotropic = return_isotropic
        self.nbh_agg = L.Aggregate(2)
        self.atom_agg = L.Aggregate(1)

        self.cutoff_network = cutoff_network

    def forward(self, inputs):
        positions = inputs[Structure.R]
        neighbors = inputs[Structure.neighbors]
        nbh_mask = inputs[Structure.neighbor_mask]
        atom_mask = inputs[Structure.atom_mask]

        # Get environment distances and positions
        distances, dist_vecs = L.atom_distances(positions, neighbors, return_vecs=True)

        # Get atomic contributions
        contributions = self.out_net(inputs)
        # c1 = contributions[:, :, 0]
        # c2 = contributions[:, :, 1]

        # norm = torch.norm(c1, dim=1)
        #
        #       c1 = c1 / norm[:, None]
        #       c2 = c2 / norm[:, None]

        # print(torch.mean(torch.sum(c1, 1)), "C1")
        # print(torch.mean(torch.sum(c2, 1)), "C2")

        # Redistribute contributions to neighbors
        # B x A x N x 1
        # neighbor_contributions = L.neighbor_elements(c1, neighbors)
        neighbor_contributions = L.neighbor_elements(contributions, neighbors)

        if self.cutoff_network is not None:
            f_cut = self.cutoff_network(distances)[..., None]
            neighbor_contributions = neighbor_contributions * f_cut

        neighbor_contributions1 = neighbor_contributions[:, :, :, 0]
        neighbor_contributions2 = neighbor_contributions[:, :, :, 1]

        # B x A x N x 3
        atomic_dipoles = self.nbh_agg(
            dist_vecs * neighbor_contributions1[..., None], nbh_mask
        )
        # B x A x N x 3

        masked_dist = (distances ** 3 * nbh_mask) + (1 - nbh_mask)
        nbh_fields = (
            dist_vecs * neighbor_contributions2[..., None] / masked_dist[..., None]
        )
        atomic_fields = self.nbh_agg(nbh_fields, nbh_mask)
        field_norm = torch.norm(atomic_fields, dim=-1, keepdim=True)
        field_norm = field_norm + (field_norm < 1e-10).float()
        atomic_fields = atomic_fields / field_norm

        atomic_polar = atomic_dipoles[..., None] * atomic_fields[:, :, None, :]

        # Symmetrize
        atomic_polar = symmetric_product(atomic_polar)

        global_polar = self.atom_agg(atomic_polar, atom_mask[..., None])

        result = {
            # "y_i": atomic_polar,
            "y": global_polar
        }

        if self.return_isotropic:
            result["y_iso"] = torch.mean(
                torch.diagonal(global_polar, dim1=-2, dim2=-1), dim=-1, keepdim=True
            )
        return result


def symmetric_product(tensor):
    """
    Symmetric outer product of tensor
    """
    shape = tensor.size()
    idx = list(range(len(shape)))
    idx[-1], idx[-2] = idx[-2], idx[-1]
    return 0.5 * (tensor + tensor.permute(*idx))


class ModelError(Exception):
    pass


class Properties:
    """
    Collection of all available model properties.
    """

    energy = "energy"
    forces = "forces"
    dipole_moment = "dipole_moment"
    total_dipole_moment = "total_dipole_moment"
    polarizability = "polarizability"
    iso_polarizability = "iso_polarizability"
    at_polarizability = "at_polarizability"
    charges = "charges"
    energy_contributions = "energy_contributions"


class PropertyModel(nn.Module):
    """

    """

    def __init__(
            self,
            output_modules
    ):
        super(PropertyModel, self).__init__()
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        self.requires_dr = any([om.dr_property for om in output_modules])

    def forward(self, input):
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(input))
        return outs


class NewAtomisticModel(nn.Module):

    def __init__(self, representation, property_model):
        super(NewAtomisticModel, self).__init__()
        self.representation = representation
        self.output_layer = property_model
        self.requires_dr = property_model.requires_dr

    def forward(self, inputs):
        if self.requires_dr:
            inputs[Structure.R].requires_grad_()
        inputs["representation"] = self.representation(inputs)
        return self.output_layer(inputs)


class EasyPropertyModel(PropertyModel):

    def __init__(
            self,
            n_in,
            properties,
            n_layers=2,
            n_neurons=None,
            activation=schnetpack.nn.activations.shifted_softplus,
            mean=None,
            stddev=None,
            atomrefs=None,
            max_z=100,
            outnet=None,
            train_embeddings=False
    ):
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.max_z = max_z
        self.outnet = outnet
        self.train_embeddings = train_embeddings
        self.mean = mean
        self.stddev = stddev
        self.atomrefs = atomrefs

        output_modules = self.get_output_modules(properties)
        super(EasyPropertyModel, self).__init__(output_modules)

    def get_output_modules(self, properties):
        """
        Build the output modules from the property dict.
        Args:
            properties (dict): property dict

        Returns (nn.ModuleList):
            list of output modules
        """
        # split by output type
        property_modules = {}
        contrib_properties = {}
        derivative_properties = {}
        for prop_name, module_type in properties.items():
            split_type = module_type.split(':')
            if len(split_type) == 1:
                property_modules[prop_name] = module_type
            elif split_type[0] == 'contrib':
                contrib_properties[split_type[1]] = prop_name
            elif split_type[0] == 'der':
                derivative_properties[split_type[1]] = prop_name
            else:
                raise NotImplementedError
        # build models
        output_models = []
        for prop_name, prop_type in property_modules.items():
            contribution_property = None
            dr_property = None
            if prop_name in contrib_properties.keys():
                contribution_property = contrib_properties[prop_name]
            if prop_name in derivative_properties.keys():
                dr_property = derivative_properties[prop_name]
            output_models.append(self._build_module(prop_type, prop_name,
                                                    contribution_property, dr_property))
        return nn.ModuleList(output_models)

    def _build_module(self, module_type, prop_name, contribution_property, dr_property):
        # todo: check create graph
        if module_type == 'atomwise':
            return Atomwise(n_in=self.n_in, property=prop_name,
                            contribution_property=contribution_property,
                            dr_property=dr_property, n_layers=self.n_layers,
                            n_neurons=self.n_neurons, activation=self.activation,
                            max_z=self.max_z, outnet=self.outnet,
                            train_embeddings=self.train_embeddings, mean=self.mean,
                            stddev=self.stddev, atomref=self.atomrefs)
        elif module_type == 'atomwise_avg':
            return Atomwise(n_in=self.n_in, property=prop_name,
                            contribution_property=contribution_property,
                            dr_property=dr_property, n_layers=self.n_layers,
                            n_neurons=self.n_neurons, activation=self.activation,
                            max_z=self.max_z, outnet=self.outnet,
                            train_embeddings=self.train_embeddings, mean=self.mean,
                            stddev=self.stddev, atomref=self.atomrefs,
                            aggregation_mode='avg')
        elif module_type == 'elementalatomwise':
            return ElementalAtomwise(
                n_in=self.n_in, aggregation_mode="sum", n_layers=self.n_layers,
                dr_property=dr_property, create_graph=False,
                elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                activation=self.activation,
                contribution_property=contribution_property,
                mean=self.mean, stddev=self.stddev, atomref=self.atomrefs,
                max_z=100, property=prop_name)
        else:
            raise NotImplementedError
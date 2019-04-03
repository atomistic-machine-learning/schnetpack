r"""
Classes for output modules.
"""

from collections import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.autograd import grad

from schnetpack.nn.base import GetItem, ScaleShift, Aggregate
from schnetpack.nn.blocks import MLP, GatedNetwork
from schnetpack.data import Structure
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.neighbors import atom_distances, neighbor_elements


class AtomisticModel(nn.Module):
    """Assemble a model from atom-wise representation module and output module(s).

    Args:
        representation(nn.Module): network block that computes atom-wise embeddings.
        output_modules(nn.OutputModule or iterable of nn.OutputModule): network block
            that predicts desired property from representation.

    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        # atom-wise representation network
        self.representation = representation
        # output network(s) computing desired property
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
        r"""Compute assembled network output.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            If `output_modules` is an instance on nn.OutputModule, predicted tensor
            is returned. If `output_modules` is a sequence of nn.OutputModule
            instances, a dictionary of {output_module: predicted_tensor} is returned.

        """
        if self.requires_dr:
            inputs[Structure.R].requires_grad_()
        # compute atom-wise embeddings & add it to inputs dictionary
        inputs["representation"] = self.representation(inputs)
        # compute network output (i.e. desired property) from atom-wise representation
        if isinstance(self.output_modules, nn.ModuleList):
            outs = {}
            for output_module in self.output_modules:
                outs[output_module] = output_module(inputs)
        else:
            outs = self.output_modules(inputs)
        return outs


class OutputModule(nn.Module):
    """Base class for output modules.

    Args:
        requires_dr (bool, optional): if True, `forward` method returns the derivative
            of the output as well.

    """

    def __init__(self, requires_dr=False):
        super(OutputModule, self).__init__()
        self.requires_dr = requires_dr

    def forward(self, inputs):
        """Define the computation performed at every call.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
        raise NotImplementedError


class Atomwise(OutputModule):
    """Predict atom-wise contributions and accumulate global prediction.

    Args:
        n_in (int, optional): input dimension of atom-wise representation.
        n_out (int, optional): output dimension of target property.
        aggregation_mode (str, optional): pooling method of atom-wise predictions.
            Options: 'sum' (sum atom-wise output) or 'avg' (average atom-wise output).
        n_layers (int, optional): total number of layers in the network.
        n_neurons (list of int, optional): number of neurons in each layer of the
            output network. If `None`, divide neurons by 2 in each layer.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
        return_contributions (bool, optional): if True, `forward` method returns
            atom-wise predictions as well.
        requires_dr (bool, optional): if True, derivative w.r.t. atom positions is
            required.
        create_graph ():
        mean (torch.FloatTensor, optional): property mean used in output `ScaleShift`.
        stddev (torch.FloatTensor, optional): property standard deviation used in
            output `ScaleShift`.
        atomref (torch.Tensor, optional): reference atom-wise properties.
            Expects an (max_z + 1) x 1 array where atomref[Z] corresponds to the
            reference property of element Z. The value of atomref[0] must be
            zero, as this corresponds to the reference property for for "mask" atoms.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
            Only relevant only if train_embeddings=True.
        outnet (callable, optional): network used for atomistic prediction which takes
            SchNetPack dictionary of input tensors. If None, a pyramidal network is
            generated automatically.
        train_embeddings (bool, optional): if True, atomref will be learned from data.

    """

    def __init__(
        self,
        n_in=128,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=shifted_softplus,
        return_contributions=False,
        requires_dr=False,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        max_z=100,
        outnet=None,
        train_embeddings=False,
    ):
        super(Atomwise, self).__init__(requires_dr)

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.return_contributions = return_contributions

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32)),
                freeze=not train_embeddings,
            )
        elif train_embeddings:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(np.zeros((max_z, 1), dtype=np.float32)),
                freeze=not train_embeddings,
            )
        else:
            self.atomref = None
        # network block
        if outnet is None:
            # assign fully connected feed-forward block as network block
            self.out_net = nn.Sequential(
                GetItem("representation"),
                MLP(n_in, n_out, n_neurons, n_layers, activation=activation),
            )
        else:
            self.out_net = outnet
        # assign mean and standard deviation of property
        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev
        # standardization layer
        self.standardize = ScaleShift(mean, stddev)
        # pooling layer
        if aggregation_mode == "sum":
            self.atom_pool = Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = Aggregate(axis=1, mean=True)
        else:
            raise ValueError("Invalid aggregation_mode={0}!".format(aggregation_mode))

    def forward(self, inputs):
        r"""Compute atom-wise outputs and aggregate them.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            dict: network results; the key "y" contains aggregated output.
                If return_contributions=True, the key "yi" would contain atom-wise
                contributions.

        """
        # get atomic number and mask
        atomic_numbers = inputs[Structure.Z]
        atom_mask = inputs[Structure.atom_mask]
        # compute atom-wise predictions and standardize them
        yi = self.out_net(inputs)
        yi = self.standardize(yi)
        # add atom-wise reference values
        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0
        # store aggregated atom-wise predictions
        result = {"y": self.atom_pool(yi, atom_mask)}
        # store atom-wise contributions to predicted value
        if self.return_contributions:
            result["yi"] = yi
        return result


class Energy(Atomwise):
    """Energy Prediction Class.

    Args:
        n_in (int): dimension of input representation.
        aggregation_mode (str, optional): Mode of aggregation. Should be one of
            - sum
            - avg
        n_layers (int, optional): number of nn in output network.
        n_neurons (list of int or None, optional): number of neurons in each layer
            of the output network. If None, divide neurons by 2 in each layer.
        activation (callable, optional): activation function for hidden nn.
        return_contributions (bool, optional): if True, latent atomic contributions
            are returned as well.
        create_graph (bool, optional): if True, graph of forces is created.
        return_force (bool, optional): if True, forces will be calculated.
        mean (torch.FloatTensor, optional): mean of energy.
        stddev (torch.FloatTensor, optional): standard deviation of the energy.
        atomref (torch.Tensor, optional): reference single-atom properties.
        max_z (int, optional):
        outnet (callable, optional): network used for atomistic outputs.
            Takes schnetpack input dictionary as input. Output is not normalized.
            If set to None, a pyramidal network is generated automatically.

    """

    def __init__(
        self,
        n_in,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=shifted_softplus,
        return_contributions=False,
        create_graph=False,
        return_force=False,
        mean=None,
        stddev=None,
        atomref=None,
        max_z=100,
        outnet=None,
    ):
        super(Energy, self).__init__(
            n_in,
            1,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation,
            return_contributions,
            return_force,
            create_graph,
            mean,
            stddev,
            atomref,
            max_z,
            outnet,
        )

    def forward(self, inputs):
        r"""
        predicts energy
        """
        result = super(Energy, self).forward(inputs)

        if self.requires_dr:
            forces = -grad(
                result["y"],
                inputs[Structure.R],
                grad_outputs=torch.ones_like(result["y"]),
                create_graph=self.create_graph,
            )[0]
            result["dydx"] = forces

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
        activation=shifted_softplus,
        return_charges=False,
        requires_dr=False,
        outnet=None,
        predict_magnitude=False,
        mean=torch.FloatTensor([0.0]),
        stddev=torch.FloatTensor([1.0]),
    ):
        self.return_charges = return_charges
        self.predict_magnitude = predict_magnitude
        super(DipoleMoment, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_neurons,
            activation=activation,
            requires_dr=requires_dr,
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
        requires_dr=False,
        create_graph=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden=50,
        activation=shifted_softplus,
        return_contributions=False,
        mean=None,
        stddev=None,
        atomref=None,
        max_z=100,
    ):
        outnet = GatedNetwork(
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
            return_contributions,
            requires_dr,
            create_graph,
            mean,
            stddev,
            atomref,
            max_z,
            outnet,
        )


class ElementalEnergy(Energy):
    """
    Predicts atom-wise contributions, but uses one separate network for every chemical element of
    the central atoms.
    Particularly useful for networks of Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        pool_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of layers in output network (default: 3)
        return_force (bool): True, if derivative w.r.t. atom positions is required (default: False)
        elements (set of int): List of atomic numbers present in the training set {1,6,7,8,9} for QM9. (default: (1,6,7,8,9)
        n_hidden (int): number of neurons in each hidden layer of the output network. (default: 50)
        activation (torch.Function): activation function for hidden nn (default: schnetpack.nn.activations.shifted_softplus)
        return_contributions (bool): If True, latent atomic con tributions are returned as well (default: False)
        mean (torch.FloatTensor): mean of energy
        stddev (torch.FloatTensor): standard deviation of energy
        atomref (torch.tensor): reference single-atom properties
        max_z (int): ignored if atomref is not learned (default: 100)
    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=3,
        return_force=False,
        create_graph=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden=50,
        activation=shifted_softplus,
        return_contributions=False,
        mean=None,
        stddev=None,
        atomref=None,
        max_z=100,
    ):
        outnet = GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalEnergy, self).__init__(
            n_in,
            aggregation_mode,
            n_layers,
            None,
            activation,
            return_contributions,
            create_graph=create_graph,
            return_force=return_force,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
            max_z=max_z,
            outnet=outnet,
        )


class ElementalDipoleMoment(DipoleMoment):
    """
    Predicts partial charges and computes dipole moment using serparate NNs for every different element.
    Particularly useful for networks of Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of representation (default: 1)
        n_layers (int): number of layers in output network (default: 3)
        return_charges (bool): If True, latent atomic charges are returned as well (default: False)
        requires_dr (bool): Set True, if derivative w.r.t. atom positions is required (default: False)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment instead of the vector (default: False)
        elements (set of int): List of atomic numbers present in the training set {1,6,7,8,9} for QM9. (default: frozenset(1,6,7,8,9))
        n_hidden (int): number of neurons in each layer of the output network. (default: 50)
        activation (function): activation function for hidden nn (default: schnetpack.nn.activations.shifted_softplus)
        activation (function): activation function for hidden nn
        mean (torch.FloatTensor): mean of energy
        stddev (torch.FloatTensor): standard deviation of energy
    """

    def __init__(
        self,
        n_in,
        n_out=1,
        n_layers=3,
        return_charges=False,
        requires_dr=False,
        predict_magnitude=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden=50,
        activation=shifted_softplus,
        mean=None,
        stddev=None,
    ):
        outnet = GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalDipoleMoment, self).__init__(
            n_in,
            n_layers,
            None,
            activation,
            return_charges,
            requires_dr,
            outnet,
            predict_magnitude,
            mean,
            stddev,
        )


class Polarizability(Atomwise):
    def __init__(
        self,
        n_in,
        pool_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=shifted_softplus,
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
        distances, dist_vecs = atom_distances(positions, neighbors, return_vecs=True)

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
        # neighbor_contributions = neighbor_elements(c1, neighbors)
        neighbor_contributions = neighbor_elements(contributions, neighbors)

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
    def __init__(
        self, n_in, properties, mean, stddev, atomrefs, cutoff_network, cutoff
    ):
        super(PropertyModel, self).__init__()

        self.n_in = n_in
        self.properties = properties

        self._init_property_flags(properties)
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
                n_in,
                aggregation_mode="sum",
                return_force=self.need_forces,
                return_contributions=self.need_energy_contributions,
                mean=mu,
                stddev=std,
                atomref=atomref,
                create_graph=True,
            )
            outputs[Properties.energy] = energy_module
            self.bias[Properties.energy] = energy_module.out_net[1].out_net[1].bias
        if self.need_dipole or self.need_total_dipole:
            dipole_module = DipoleMoment(
                n_in, predict_magnitude=self.need_total_dipole, return_charges=True
            )
            if self.need_dipole:
                outputs[Properties.dipole_moment] = dipole_module
            else:
                outputs[Properties.total_dipole_moment] = dipole_module
        if self.need_polarizability or self.need_iso_polarizability:
            polar_module = Polarizability(
                n_in,
                return_isotropic=self.need_iso_polarizability,
                cutoff_network=self.cutoff_network,
            )
            outputs[Properties.polarizability] = polar_module

        self.output_dict = nn.ModuleDict(outputs)

    def _init_property_flags(self, properties):
        self.need_energy = Properties.energy in properties
        self.need_forces = Properties.forces in properties
        self.need_dipole = Properties.dipole_moment in properties
        self.need_total_dipole = Properties.total_dipole_moment in properties
        self.need_polarizability = Properties.polarizability in properties
        self.need_at_polarizability = Properties.at_polarizability in properties
        self.need_iso_polarizability = Properties.iso_polarizability in properties
        self.need_energy_contributions = Properties.energy_contributions in properties

    def forward(self, inputs):
        outs = {}
        for prop, mod in self.output_dict.items():
            o = mod(inputs)
            outs[prop] = o["y"]
            if prop == Properties.energy and self.need_forces:
                outs[Properties.forces] = o["dydx"]
            if prop == Properties.energy and self.need_energy_contributions:
                outs[Properties.energy_contributions] = o["yi"]
            if prop in [Properties.dipole_moment, Properties.total_dipole_moment]:
                outs[Properties.charges] = o["yi"]
            if prop == Properties.polarizability:
                if self.need_iso_polarizability:
                    outs[Properties.iso_polarizability] = o["y_iso"]
                if self.need_at_polarizability:
                    outs[Properties.at_polarizability] = o["y_i"]
        return outs

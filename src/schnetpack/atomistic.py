r"""
Classes for output modules.
"""

from collections import Iterable
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

import schnetpack.nn as L
import schnetpack.nn.activations
import schnetpack.nn.base
import schnetpack.nn.blocks
from schnetpack.data import Structure


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
        inputs['representation'] = self.representation(inputs)

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
        n_in (int): input dimension
        n_out (int): output dimension
        requires_dr (bool): specifies if the derivative of the ouput is required
    """

    def __init__(self, n_in, n_out, requires_dr=False):
        self.n_in = n_in
        self.n_out = n_out
        self.requires_dr = requires_dr
        super(OutputModule, self).__init__()

    def forward(self, inputs):
        r"""
        Should be overwritten
        """
        raise NotImplementedError


atomwise = namedtuple('atomwise', ['y', 'yi', 'grad'])


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
        atomref (torch.Tensor): reference single-atom properties
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

    def __init__(self, n_in=128, n_out=1, pool_mode='sum', n_layers=2, n_neurons=None,
                 activation=schnetpack.nn.activations.shifted_softplus, return_contributions=False,
                 requires_dr=False, create_graph=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
                 train_embeddings=False):
        super(Atomwise, self).__init__(n_in, n_out, requires_dr)

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.return_contributions = return_contributions

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(torch.from_numpy(atomref.astype(np.float32)),
                                                        freeze=train_embeddings)
        elif train_embeddings:
            self.atomref = nn.Embedding.from_pretrained(torch.from_numpy(np.zeros((max_z, 1), dtype=np.float32)),
                                                        freeze=train_embeddings)
        else:
            self.atomref = None

        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem('representation'),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation)
            )
        else:
            self.out_net = outnet

        # Make standardization separate
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        if pool_mode == 'sum':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif pool_mode == 'avg':
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

        result = (y,)

        if self.return_contributions:
            result = result + (yi,)

        return result


class Energy(Atomwise):
    """
        Predicts energy.

        Args:
            n_in (int): input dimension of representation
            pool_mode (str): one of {sum, avg} (default: sum)
            n_layers (int): number of nn in output network (default: 2)
            n_neurons (list of int or None): number of neurons in each layer of the output network.
                                              If `None`, divide neurons by 2 in each layer. (default: none)
            activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
            return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
            create_graph (bool): if True, graph of forces is created (default: False)
            return_force (bool): if True, forces will be calculated (default: False)
            mean (torch.FloatTensor): mean of energy (default: None)
            stddev (torch.FloatTensor): standard deviation of the energy (default: None)
            atomref (torch.Tensor): reference single-atom properties
            outnet (callable): network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                               not normalized. If set to None (default), a pyramidal network is generated automatically.

        Returns:
            tuple: Prediction for energy.

            If requires_dr is true additionally returns forces
        """
    def __init__(self, n_in, pool_mode='sum', n_layers=2, n_neurons=None,
                 activation=schnetpack.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=False,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None):
        super(Energy, self).__init__(n_in, 1, pool_mode, n_layers, n_neurons, activation,
                                     return_contributions, return_force, create_graph, mean, stddev,
                                     atomref, 100, outnet)

    def forward(self, inputs):
        r"""
        predicts energy
        """
        result = super(Energy, self).forward(inputs)

        if self.requires_dr:
            forces = -grad(result[0], inputs[Structure.R],
                           grad_outputs=torch.ones_like(result[0]),
                           create_graph=self.create_graph)[0]
            result = result + (forces,)

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

        If predict_magnitute is true returns the magnitude of the dipole moment instead of the vector

        If return_charges is true returns either vector or magnitude of the dipole moment, and latent atomic charges

        If requires_dr is true returns derivative w.r.t. atom positions
    """

    def __init__(self, n_in, n_layers=2, n_neurons=None, activation=schnetpack.nn.activations.shifted_softplus,
                 return_charges=False, requires_dr=False, outnet=None, predict_magnitude=False,
                 mean=torch.FloatTensor([0.0]), stddev=torch.FloatTensor([1.0])):
        self.return_charges = return_charges
        self.predict_magnitude = predict_magnitude
        super(DipoleMoment, self).__init__(n_in, 1, 'sum', n_layers, n_neurons, activation=activation,
                                           requires_dr=requires_dr, mean=mean, stddev=stddev, outnet=outnet)

    def forward(self, inputs):
        """
        predicts dipole moment
        """
        positions = inputs[Structure.R]
        atom_mask = inputs[Structure.atom_mask][:, :, None]

        charges = self.out_net(inputs)
        yi = positions * charges * atom_mask
        y = self.atom_pool(yi)

        if self.predict_magnitude:
            result = (torch.norm(y, dim=1, keepdim=True),)
        else:
            result = (y,)

        if self.return_charges:
            result = result + (yi,)

        return result


class ElementalAtomwise(Atomwise):
    """
    Predicts properties in atom-wise fashion using a separate network for every chemical element of the central atom.
    Particularly useful for networks of Behler-Parrinello type.
    """

    def __init__(self, n_in, n_out=1, pool_mode='sum', n_layers=3, requires_dr=False, create_graph=False,
                 elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                 activation=schnetpack.nn.activations.shifted_softplus,
                 return_contributions=False, mean=None, stddev=None, atomref=None, max_z=100):
        outnet = schnetpack.nn.blocks.GatedNetwork(n_in, n_out, elements, n_hidden=n_hidden, n_layers=n_layers,
                                                   activation=activation)

        super(ElementalAtomwise, self).__init__(n_in, n_out, pool_mode, n_layers, None, activation,
                                                return_contributions, requires_dr, create_graph, mean, stddev, atomref,
                                                max_z, outnet)


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

    def __init__(self, n_in, n_out=1, pool_mode='sum', n_layers=3, return_force=False, create_graph=False,
                 elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                 activation=schnetpack.nn.activations.shifted_softplus, return_contributions=False, mean=None,
                 stddev=None, atomref=None, max_z=100):
        outnet = schnetpack.nn.blocks.GatedNetwork(n_in, n_out, elements, n_hidden=n_hidden, n_layers=n_layers,
                                                   activation=activation)

        super(ElementalEnergy, self).__init__(n_in, pool_mode, n_layers, None, activation, return_contributions,
                                              create_graph=create_graph, return_force=return_force, mean=mean,
                                              stddev=stddev, atomref=atomref, max_z=max_z, outnet=outnet)


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

    def __init__(self, n_in, n_out=1, n_layers=3, return_charges=False, requires_dr=False, predict_magnitude=False,
                 elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                 activation=schnetpack.nn.activations.shifted_softplus,
                 mean=None, stddev=None):
        outnet = schnetpack.nn.blocks.GatedNetwork(n_in, n_out, elements, n_hidden=n_hidden, n_layers=n_layers,
                                                   activation=activation)

        super(ElementalDipoleMoment, self).__init__(n_in, n_layers, None, activation, return_charges, requires_dr,
                                                    outnet, predict_magnitude, mean, stddev)

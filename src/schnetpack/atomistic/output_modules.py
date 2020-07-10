import numpy as np
import torch
from torch import nn as nn
from torch.autograd import grad

import schnetpack as spk
from schnetpack import nn as L, Properties

__all__ = [
    "Atomwise",
    "ElementalAtomwise",
    "DipoleMoment",
    "ElementalDipoleMoment",
    "Polarizability",
    "ElectronicSpatialExtent",
    "AtomwiseCorrected",
]


class AtomwiseError(Exception):
    pass


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        stress (str or None): Name of stress property. Compute the derivative with
            respect to the cell parameters if not None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        out_net (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=spk.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=True,
        mean=None,
        stddev=None,
        atomref=None,
        out_net=None,
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref[property].astype(np.float32))
            )
        else:
            self.atomref = None

        # build output network
        if out_net is None:
            self.out_net = nn.Sequential(
                spk.nn.base.GetItem("representation"),
                spk.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = out_net

        # build standardization layer
        self.standardize = spk.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = spk.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = spk.nn.base.Aggregate(axis=1, mean=True)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs):
        r"""
        Predicts atomwise properties.

        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi, atom_mask)

        # collect results
        result = {self.property: y}

        if self.contributions is not None:
            result[self.contributions] = yi

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

        return result


class DipoleMoment(Atomwise):
    """
    Predicts latent partial charges and calculates dipole moment.

    Args:
        n_in (int): input dimension of representation
        n_layers (int): number of layers in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (torch.Function): activation function for hidden nn
            (default: schnetpack.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment
            instead of the vector (default: False)
        mean (torch.FloatTensor or None): mean of dipole (default: None)
        stddev (torch.FloatTensor or None): stddev of dipole (default: None)

    Returns:
        dict: vector for the dipole moment

        If predict_magnitude is True returns the magnitude of the dipole moment
        instead of the vector.

        If contributions is not None latent atomic charges are added to the output
        dictionary.
    """

    def __init__(
        self,
        n_in,
        n_layers=2,
        n_neurons=None,
        activation=spk.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        predict_magnitude=False,
        mean=None,
        stddev=None,
        out_net=None,
    ):
        self.predict_magnitude = predict_magnitude
        super(DipoleMoment, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            out_net=out_net,
            property=property,
            contributions=contributions,
        )

    def forward(self, inputs):
        """
        predicts dipole moment
        """
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask][:, :, None]

        # run prediction
        charges = self.out_net(inputs) * atom_mask
        yi = positions * charges
        y = self.atom_pool(yi)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=True)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = charges

        return result


class ElementalAtomwise(Atomwise):
    """
    Predicts properties in atom-wise fashion using a separate network for every chemical
    element of the central atom. Particularly useful for networks of
    Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation (default: 128)
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 3)
        property (str): name of the output property (default: "y")
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        elements (set of int): List of atomic numbers present in the training set
            {1,6,7,8,9} for QM9. (default: frozenset(1,6,7,8,9))
        n_hidden (int): number of neurons in each layer of the output network.
            (default: 50)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
    """

    def __init__(
        self,
        n_in=128,
        n_out=1,
        aggregation_mode="sum",
        n_layers=3,
        property="y",
        derivative=None,
        negative_dr=False,
        contributions=None,
        create_graph=True,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden=50,
        activation=spk.nn.activations.shifted_softplus,
        mean=None,
        stddev=None,
        atomref=None,
    ):
        out_net = spk.nn.blocks.GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalAtomwise, self).__init__(
            n_in=n_in,
            n_out=n_out,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=None,
            activation=activation,
            property=property,
            contributions=contributions,
            derivative=derivative,
            negative_dr=negative_dr,
            create_graph=create_graph,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
            out_net=out_net,
        )


class ElementalDipoleMoment(DipoleMoment):
    """
    Predicts partial charges and computes dipole moment using serparate NNs for every different element.
    Particularly useful for networks of Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of representation (default: 1)
        n_layers (int): number of layers in output network (default: 3)
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
        contributions=False,
        property="y",
        predict_magnitude=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden=50,
        activation=spk.nn.activations.shifted_softplus,
        mean=None,
        stddev=None,
    ):
        out_net = spk.nn.blocks.GatedNetwork(
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
            activation=activation,
            property=property,
            contributions=contributions,
            out_net=out_net,
            predict_magnitude=predict_magnitude,
            mean=mean,
            stddev=stddev,
        )


class Polarizability(Atomwise):
    """
    Predicts polarizability of input molecules.

    Args:
        n_in (int): input dimension of representation (default: 128)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        isotropic (bool): return isotropic polarizability if True. (default: False)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        out_net (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)
        cutoff_network (nn.Module): cutoff network (default: None)

    Returns:
        dict: Polarizability of molecules

        Adds isotropic polarizability if isotropic is not None.
    """

    def __init__(
        self,
        n_in=128,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=L.shifted_softplus,
        property="y",
        isotropic=False,
        create_graph=True,
        out_net=None,
        cutoff_network=None,
    ):
        super(Polarizability, self).__init__(
            n_in=n_in,
            n_out=2,
            n_layers=n_layers,
            aggregation_mode=aggregation_mode,
            n_neurons=n_neurons,
            activation=activation,
            property=property,
            derivative=None,
            create_graph=create_graph,
            out_net=out_net,
        )
        self.isotropic = isotropic
        self.nbh_agg = L.Aggregate(2)
        self.atom_agg = L.Aggregate(1)

        self.cutoff_network = cutoff_network

    def forward(self, inputs):
        positions = inputs[Properties.R]
        neighbors = inputs[Properties.neighbors]
        nbh_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # Get environment distances and positions
        distances, dist_vecs = L.atom_distances(positions, neighbors, return_vecs=True)

        # Get atomic contributions
        contributions = self.out_net(inputs)

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
            self.property: global_polar
        }

        if self.isotropic:
            result[self.property] = torch.mean(
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


class ElectronicSpatialExtent(Atomwise):
    """
    Predicts the electronic spatial extent using a formalism close to the dipole moment layer.
    The electronic spatial extent is discretized as a sum of atomic latent contributions
    weighted by the squared distance of the atom from the center of mass (SchNetPack default).

    .. math:: ESE = \sum_i^N | R_{i0} |^2 q(R)

    Args:
        n_in (int): input dimension of representation
        n_layers (int): number of layers in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (torch.Function): activation function for hidden nn
            (default: schnetpack.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        mean (torch.FloatTensor or None): mean of dipole (default: None)
        stddev (torch.FloatTensor or None): stddev of dipole (default: None)

    Returns:
        dict: the electronic spatial extent

        If contributions is not None latent atomic charges are added to the output
        dictionary.
    """

    def __init__(
        self,
        n_in,
        n_layers=2,
        n_neurons=None,
        activation=spk.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        mean=None,
        stddev=None,
        out_net=None,
    ):
        super(ElectronicSpatialExtent, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            out_net=out_net,
            property=property,
            contributions=contributions,
        )

    def forward(self, inputs):
        """
        Predicts the electronic spatial extent.
        """
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask][:, :, None]

        # run prediction
        charges = self.out_net(inputs) * atom_mask
        yi = torch.norm(positions, 2, 2, keepdim=True) ** 2 * charges
        y = self.atom_pool(yi)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = charges

        return result


class AtomwiseCorrected(Atomwise):
    r"""
    Predicts atom-wise contributions and accumulates global prediction. Similar to
    Atomwise but with the support of correction terms. Deduced from PhysNet.

    Args:
        n_in (int): input dimension
        n_out (int, optional): dimension of output property
        n_layers (int, optional): number of layers for out_net if out_net is not
            specified
        n_neurons (int, optional): number of neurons for out_net if out_net is not
            specified
        out_net (nn.Module, optional): custom output module. If nothing is specified,
            the out-net is build with the use of n_layers, n_neurons and n_out.
        activation (callable, optional): activation function
        charge_net (nn.Module, optional): custom charge module. If nothing is
            specified, the charge-net is build with the use of n_layers, n_neurons and
            n_out.
        property (str, optional): name of property in output dict
        contributions (str, optional): returns property contributions if not None
        derivative (str, optional): returns derivative if not None
        negative_dr (bool, optional): multiply derivative with -1 if selected
        # mean=None,
        # stddev=None,
        # atomref=None,
        aggregation_mode (str, optional): mode of aggregation; 'sum' or 'mean'
        corrections (iterable, optional): collection of correction modules
        use_element_bias (bool, optional): makes use trainable element-bias
        max_z (int, optional): maximum atomic number for element bias layer

    """
    # todo: hier allgemein lassen, oder immer qi, ei, ... zurückgeben?
    def __init__(
        self,
        n_in,
        n_out=1,
        n_layers=1,
        n_neurons=None,
        out_net=None,
        activation=spk.nn.Swish,
        charge_net=None,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=True,
        # todo: negative_dr: bool or float?
        dipole_moment=None,
        # mean=None,
        # stddev=None,
        # atomref=None,
        aggregation_mode="sum",
        corrections=[],
        use_element_bias=False,
        max_z=87,
    ):
        if out_net is None:
            out_net = nn.Sequential(
                spk.nn.base.GetItem("representation"),
                spk.nn.Dense(n_in, n_out, bias=False),
            )
        super(AtomwiseCorrected, self).__init__(
            n_in=n_in,
            n_out=n_out,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            property=property,
            contributions=contributions,
            derivative=derivative,
            negative_dr=negative_dr,
            stress=None,
            create_graph=True,
            # todo: eventually support mean and atomref?
            mean=None,
            stddev=None,
            atomref=None,
            out_net=out_net,
        )

        # initialize attributes
        self.use_element_bias = use_element_bias
        self.dipole_moment = dipole_moment

        # define modules
        # element bias
        self.element_bias = nn.Parameter(torch.Tensor(max_z))

        # charge predicting network
        if charge_net is None:
            charge_net = nn.Sequential(
                spk.nn.base.GetItem("representation"),
                spk.nn.Dense(n_in, n_out, bias=False),
            )
        self.charge_net = charge_net

        # correction terms
        self.corrections = corrections

    def forward(self, inputs):
        r"""
        Predicts atomwise properties.

        """
        # get input properties
        total_charges = inputs[Properties.charges]
        positions = inputs[Properties.position]
        atomic_numbers = inputs[Properties.atomic_numbers]
        atom_mask = inputs[Properties.atom_mask]

        # compute atom-wise properties
        yi = self.out_net(inputs) + self.element_bias[atomic_numbers].unsqueeze(-1)
        yi = yi * atom_mask.unsqueeze(-1)

        # compute atomwise charges and charge correction
        qi = self.charge_net(inputs)
        qi = qi * atom_mask.unsqueeze(-1)

        charge_correction = total_charges - qi.sum(1)
        qi = qi + (charge_correction / atom_mask.sum(-1).unsqueeze(-1)).unsqueeze(-1)

        # collect predictions
        atomwise_predictions = dict(yi=yi, qi=qi)

        # compute corrections
        # todo: recompute r_ij if different cutoffs
        y_corr = 0.0
        for correction_layer in self.corrections:
            y_corr += correction_layer(inputs, atomwise_predictions)

        # compute property
        y = self.atom_pool(yi) + y_corr

        # collect results
        result = {self.property: y, "charges": qi}

        if self.contributions is not None:
            result[self.contributions] = yi

        if self.dipole_moment is not None:
            result[self.dipole_moment] = torch.sum(qi * inputs, -2)

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                positions,
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        return result

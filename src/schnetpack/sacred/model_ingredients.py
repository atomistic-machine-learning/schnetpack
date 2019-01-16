from sacred import Ingredient

from schnetpack.representation.schnet import SchNet
from schnetpack.atomistic import AtomisticModel, ModelError, Properties, \
    PropertyModel
from schnetpack.nn.cutoff import *


model_ingredient = Ingredient('model')


@model_ingredient.config
def cfg():
    """
    Base configuration for the model.

    """
    name = None
    n_atom_basis = -1


@model_ingredient.named_config
def schnet():
    """
    Default configuration for the use of the SchNet model.

    """
    name = 'schnet'

    n_atom_basis = 128
    n_filters = 128
    n_interactions = 6
    cutoff = 5.0
    n_gaussians = 25
    normalize_filter = False
    coupled_interactions = False
    max_z = 100
    cutoff_network = 'hard'
    additional_outputs = [Properties.dipole_moment]


@model_ingredient.capture
def build_model(mean, stddev, model_properties, atomrefs, additional_outputs,
                n_atom_basis, name, cutoff):
    """
    Build the model from a given config.

    Args:
        mean (dict): means of the dataset properties
        stddev (dict): stds of the dataset properties
        model_properties (list): properties for the model
        atomrefs: atomic reference data
        additional_outputs (list): additional model output that is not
            back-propagated
        n_atom_basis (int): number of features used to describe atomic
            environments
        name (str): choose model representation
        cutoff (float): cutoff radius of filters

    Returns:
        Model object
    """
    if name == 'schnet':
        representation = build_schnet(return_intermediate=False)
    else:
        raise ModelError(
            'Unknown model: {:s}'.format(name))

    cutoff_function = get_cutoff()
    output = PropertyModel(n_atom_basis, model_properties + additional_outputs,
                           mean, stddev, atomrefs,
                           cutoff_network=cutoff_function,
                           cutoff=cutoff)
    model = AtomisticModel(representation, output)
    return model


@model_ingredient.capture
def get_cutoff(cutoff_network):
    """
    Get the cutoff network.

    Args:
        cutoff_network: name of cutoff network

    Returns:
        Cutoff network object

    """
    if cutoff_network == 'hard':
        cutoff_function = HardCutoff
    elif cutoff_network == 'cosine':
        cutoff_function = CosineCutoff
    elif cutoff_network == 'mollifier':
        cutoff_function = MollifierCutoff
    else:
        raise ModelError(
            'Unrecognized cutoff {:s}'.format(cutoff_network))
    return cutoff_function


@model_ingredient.capture
def build_schnet(return_intermediate,
                 n_atom_basis, n_filters, n_interactions, cutoff,
                 n_gaussians, normalize_filter,
                 coupled_interactions, max_z):
    """
    Build and return SchNet object.

    Args:
        return_intermediate (bool): if true, also return intermediate feature
            representations after each interaction block
        n_atom_basis (int): number of features used to describe atomic
            environments
        n_filters (int): number of filters used in continuous-filter convolution
        n_interactions (int): number of interaction blocks
        cutoff (float): cutoff radius of filters
        n_gaussians (int): number of Gaussians which are used to expand atom
            distances
        normalize_filter (bool): if true, divide filter by number of neighbors
            over which convolution is applied
        coupled_interactions (bool): if true, share the weights across
            interaction blocks and filter-generating networks.
        max_z (int): maximum allowed nuclear charge in dataset. This determines
            the size of the embedding matrix.

    Returns:
        SchNet object
    """

    cutoff_function = get_cutoff()
    return SchNet(
        n_atom_basis=n_atom_basis,
        n_filters=n_filters,
        n_interactions=n_interactions,
        cutoff=cutoff,
        n_gaussians=n_gaussians,
        normalize_filter=normalize_filter,
        coupled_interactions=coupled_interactions,
        return_intermediate=return_intermediate,
        max_z=max_z,
        cutoff_network=cutoff_function,
        charged_systems=False
    )


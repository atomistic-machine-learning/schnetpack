from sacred import Ingredient

from schnetpack.representation.schnet import SchNet
from schnetpack.atomistic import AtomisticModel, ModelError, Properties, \
    PropertyModel
from schnetpack.nn.cutoff import *


model_ingredient = Ingredient('model')


@model_ingredient.config
def cfg():
    name = None
    n_atom_basis = -1


@model_ingredient.named_config
def schnet():
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


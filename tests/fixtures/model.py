import pytest
import numpy as np
from schnetpack.nn.cutoff import HardCutoff
import schnetpack as spk

__all__ = [
    # spk.representation
    "n_atom_basis",
    "n_filters",
    "n_interactions",
    "cutoff",
    "n_gaussians",
    "normalize_filter",
    "coupled_interactions",
    "return_intermediate",
    "max_z",
    "cutoff_network",
    "trainable_gaussians",
    "distance_expansion",
    "charged_systems",
    "schnet",
    "n_spatial_basis",
    "schnet_interaction",
    # spk.atomistic
    "properties1",
    "properties2",
    "output_module_1",
    "output_module_2",
    "output_modules",
    "atomistic_model",
    # spk.nn
    "gaussion_smearing_layer",
    "cfconv_layer",
    "dense_layer",
    "mlp_layer",
    "n_mlp_tiles",
    "tiled_mlp_layer",
    "elements",
    "elemental_gate_layer",
    "cutoff_layer",
    "atom_distances",
]


# spk.representation
## settings
@pytest.fixture
def n_atom_basis():
    return 128


@pytest.fixture
def n_filters():
    return 128


@pytest.fixture
def n_interactions():
    return 3


@pytest.fixture
def cutoff():
    return 5.0


@pytest.fixture
def n_gaussians():
    return 25


@pytest.fixture
def normalize_filter():
    return False


@pytest.fixture
def coupled_interactions():
    return False


@pytest.fixture
def return_intermediate():
    return False


@pytest.fixture
def max_z():
    return 100


@pytest.fixture
def cutoff_network():
    return HardCutoff


@pytest.fixture(params=[True, False])
def trainable_gaussians(request):
    return request.param


@pytest.fixture
def distance_expansion():
    return None


@pytest.fixture
def charged_systems():
    return False


## models
@pytest.fixture
def schnet(
    n_atom_basis,
    n_filters,
    n_interactions,
    cutoff,
    n_gaussians,
    normalize_filter,
    coupled_interactions,
    return_intermediate,
    max_z,
    cutoff_network,
    trainable_gaussians,
    distance_expansion,
    charged_systems,
):
    return spk.SchNet(
        n_atom_basis=n_atom_basis,
        n_filters=n_filters,
        n_interactions=n_interactions,
        cutoff=cutoff,
        n_gaussians=n_gaussians,
        normalize_filter=normalize_filter,
        coupled_interactions=coupled_interactions,
        return_intermediate=return_intermediate,
        max_z=max_z,
        cutoff_network=cutoff_network,
        trainable_gaussians=trainable_gaussians,
        distance_expansion=distance_expansion,
        charged_systems=charged_systems,
    )


@pytest.fixture
def n_spatial_basis(n_gaussians):
    return n_gaussians


@pytest.fixture
def schnet_interaction(
    n_atom_basis, n_spatial_basis, n_filters, cutoff, cutoff_network, normalize_filter
):
    return spk.representation.SchNetInteraction(
        n_atom_basis=n_atom_basis,
        n_spatial_basis=n_spatial_basis,
        n_filters=n_filters,
        cutoff=cutoff,
        cutoff_network=cutoff_network,
        normalize_filter=normalize_filter,
    )


# spk.atomistic
@pytest.fixture
def properties1(available_properties):
    return [prop for prop in available_properties if prop.endswith("1")]


@pytest.fixture
def properties2(available_properties):
    return [prop for prop in available_properties if prop.endswith("2")]


@pytest.fixture
def output_module_1(n_atom_basis, properties1):
    om_properties = get_module_properties(properties1)
    return spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        property=om_properties["property"],
        contributions=om_properties["contributions"],
        derivative=om_properties["derivative"],
    )


@pytest.fixture
def output_module_2(n_atom_basis, properties2):
    om_properties = get_module_properties(properties2)
    return spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        property=om_properties["property"],
        contributions=om_properties["contributions"],
        derivative=om_properties["derivative"],
    )


@pytest.fixture
def output_modules(output_module_1, output_module_2):
    return [output_module_1, output_module_2]


@pytest.fixture
def atomistic_model(schnet, output_modules):
    return spk.AtomisticModel(schnet, output_modules)


# spk.nn
@pytest.fixture
def gaussion_smearing_layer(n_gaussians, trainable_gaussians):
    return spk.nn.GaussianSmearing(
        n_gaussians=n_gaussians, trainable=trainable_gaussians
    )


@pytest.fixture
def cfconv_layer(n_atom_basis, n_filters, schnet_interaction, cutoff_layer):
    return spk.nn.CFConv(
        n_in=n_atom_basis,
        n_filters=n_filters,
        n_out=n_atom_basis,
        filter_network=schnet_interaction.filter_network,
        cutoff_network=cutoff_layer,
        activation=None,
        normalize_filter=False,
        axis=2,
    )


@pytest.fixture
def dense_layer(random_input_dim, random_output_dim):
    return spk.nn.Dense(random_input_dim, random_output_dim)


@pytest.fixture
def mlp_layer(random_input_dim, random_output_dim):
    print(random_input_dim, "MLP", random_output_dim)
    return spk.nn.MLP(random_input_dim, random_output_dim)


@pytest.fixture
def n_mlp_tiles():
    return np.random.randint(1, 6, 1).item()


@pytest.fixture
def tiled_mlp_layer(random_input_dim, random_output_dim, n_mlp_tiles):
    return spk.nn.TiledMultiLayerNN(random_input_dim, random_output_dim, n_mlp_tiles)


@pytest.fixture
def elements():
    return list(set(np.random.randint(1, 30, 10)))


@pytest.fixture
def elemental_gate_layer(elements):
    return spk.nn.ElementalGate(elements=elements)


@pytest.fixture
def cutoff_layer(cutoff_network, cutoff):
    return cutoff_network(cutoff=cutoff)


@pytest.fixture
def atom_distances():
    return spk.nn.AtomDistances()


# utility functions
def get_module_properties(properties):
    """
    Get dict of properties for output module.

    Args:
        properties (list): list of properties

    Returns:
        (dict): dict with prop, der and contrib

    """
    module_props = dict(property=None, derivative=None, contributions=None)
    for prop in properties:
        if prop.startswith("property"):
            module_props["property"] = prop
        elif prop.startswith("derivative"):
            module_props["derivative"] = prop
        elif prop.startswith("contributions"):
            module_props["contributions"] = prop
    return module_props

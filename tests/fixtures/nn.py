import numpy as np
import pytest

import schnetpack as spk


__all__ = [
    # nn.acsf
    "gaussion_smearing_layer",
    # nn.activations
    "swish",
    "shifted_softplus",
    "activation_function",
    # nn.base
    "dense_layer",
    "atom_distances",
    # nn.blocks
    "mlp_layer",
    "n_mlp_tiles",
    "tiled_mlp_layer",
    "elements",
    "elemental_gate_layer",
    "residual_block",
    "n_residuals",
    "residual_stack",
    # nn.cfconv
    "cfconv_layer",
    # nn.cutoff
    "cutoff_layer",
    # nn.initializers
    # nn.neighbors
]


# nn.acsf
@pytest.fixture
def gaussion_smearing_layer(n_gaussians, trainable_gaussians):
    return spk.nn.GaussianSmearing(
        n_gaussians=n_gaussians, trainable=trainable_gaussians
    )


# nn.activations
@pytest.fixture
def swish(random_input_dim):
    return spk.nn.Swish(random_input_dim)


@pytest.fixture
def shifted_softplus():
    return spk.nn.shifted_softplus


@pytest.fixture(params=[spk.nn.shifted_softplus, spk.nn.Swish])
def activation_function(request):
    return request.param


# nn.base
@pytest.fixture
def dense_layer(random_input_dim, random_output_dim):
    return spk.nn.Dense(random_input_dim, random_output_dim)


@pytest.fixture
def atom_distances():
    return spk.nn.AtomDistances()


# nn.blocks
@pytest.fixture
def mlp_layer(random_input_dim, random_output_dim):
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
def residual_block(random_input_dim, activation_function):
    return spk.nn.ResidualBlock(random_input_dim, activation=activation_function)


@pytest.fixture
def n_residuals():
    return np.random.randint(2, 10)


@pytest.fixture
def residual_stack(n_residuals, random_input_dim, activation_function):
    return spk.nn.ResidualStack(
        n_blocks=n_residuals,
        n_features=random_input_dim,
        activation=activation_function,
    )


# nn.cfconv
@pytest.fixture
def cfconv_layer(
    n_atom_basis, n_filters, schnet_interaction, cutoff_layer,
):
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


# nn.cutoff
@pytest.fixture
def cutoff_layer(cutoff_network, cutoff):
    return cutoff_network(cutoff=cutoff)


# nn.initializers


# nn.neighbors

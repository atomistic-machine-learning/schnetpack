import pytest
from schnetpack.nn.cutoff import HardCutoff
import schnetpack as spk
from .data import *

__all__ = [
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
    "output_module_1",
    "output_module_2",
    "output_modules",
    "atomistic_model",
]


# representation
## schnet
@pytest.fixture(scope="session")
def n_atom_basis():
    return 128


@pytest.fixture(scope="session")
def n_filters():
    return 128


@pytest.fixture(scope="session")
def n_interactions():
    return 1


@pytest.fixture(scope="session")
def cutoff():
    return 5.0


@pytest.fixture(scope="session")
def n_gaussians():
    return 25


@pytest.fixture(scope="session")
def normalize_filter():
    return False


@pytest.fixture(scope="session")
def coupled_interactions():
    return False


@pytest.fixture(scope="session")
def return_intermediate():
    return False


@pytest.fixture(scope="session")
def max_z():
    return 100


@pytest.fixture(scope="session")
def cutoff_network():
    return HardCutoff


@pytest.fixture(scope="session")
def trainable_gaussians():
    return False


@pytest.fixture(scope="session")
def distance_expansion():
    return None


@pytest.fixture(scope="session")
def charged_systems():
    return False


@pytest.fixture(scope="session")
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


# output modules
@pytest.fixture(scope="session")
def output_module_1(n_atom_basis, properties1):
    return spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        property=properties1[0],
        contributions=properties1[2],
        derivative=properties1[1],
    )


@pytest.fixture(scope="session")
def output_module_2(n_atom_basis, properties2):
    return spk.atomistic.Atomwise(
        n_in=n_atom_basis, property=properties2[0], derivative=properties2[1]
    )


@pytest.fixture(scope="session")
def output_modules(output_module_1, output_module_2):
    return [output_module_1, output_module_2]


@pytest.fixture(scope="session")
def atomistic_model(schnet, output_modules):
    return spk.AtomisticModel(schnet, output_modules)

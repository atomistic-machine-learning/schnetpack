import pytest
from schnetpack.nn.cutoff import HardCutoff
from schnetpack.representation import SchNet


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
]


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
        cutoff_network=cutoff_network,
        trainable_gaussians=trainable_gaussians,
        distance_expansion=distance_expansion,
        charged_systems=charged_systems,
    )

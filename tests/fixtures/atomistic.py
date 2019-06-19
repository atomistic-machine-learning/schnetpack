import pytest
from schnetpack.output_modules import Atomwise
from schnetpack.atomistic import AtomisticModel


__all__ = [
    "properties",
    "energy_module",
    "property_output",
    "result_shapes",
    "atomistic_model",
]


@pytest.fixture(scope="session")
def properties():
    return dict(
        energy="atomwise",
        forces="der:energy",
        energy_contributions="contrib:energy",
        prop="atomwise",
        prop_der="der:prop",
    )


@pytest.fixture(scope="session")
def energy_module(n_atom_basis):
    return Atomwise(
        n_in=n_atom_basis,
        property="energy",
        contributions="energy_contributions",
        derivative="forces",
    )


@pytest.fixture(scope="session")
def property_output(n_atom_basis):
    return Atomwise(n_in=n_atom_basis, property="prop", derivative="der")


@pytest.fixture(scope="session")
def result_shapes(batch_size, n_atoms):
    return dict(
        energy=[batch_size, 1],
        forces=[batch_size, n_atoms, 3],
        energy_contributions=[batch_size, n_atoms, 1],
        prop=[batch_size, 1],
        der=[batch_size, n_atoms, 3],
    )


@pytest.fixture(scope="session")
def atomistic_model(schnet, energy_module, property_output):
    return AtomisticModel(schnet, output_modules=[energy_module, property_output])

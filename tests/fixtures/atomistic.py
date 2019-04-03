import pytest
from schnetpack.atomistic import EasyPropertyModel, NewAtomisticModel


@pytest.fixture(scope='session')
def properties():
    return dict(energy='atomwise',
                forces='der:energy',
                energy_contributions='contrib:energy',
                prop='atomwise',
                prop_der='der:prop')


@pytest.fixture(scope='session')
def result_shapes(batch_size, n_atoms):
    return dict(energy=[batch_size, 1],
                forces=[batch_size, n_atoms, 3],
                energy_contributions=[batch_size, n_atoms, 1],
                prop=[batch_size, 1],
                prop_der=[batch_size, n_atoms, 3])


@pytest.fixture(scope='session')
def property_model(n_atom_basis, properties):
    return EasyPropertyModel(n_in=n_atom_basis, properties=properties)


@pytest.fixture(scope='session')
def new_atomistic_model(schnet, property_model):
    return NewAtomisticModel(schnet, property_model)
